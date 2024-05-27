//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

// -------------
// pure C++ code
// -------------
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "tpu_mlir/InitAll.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/ModuleInterpreter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace tpu_mlir;

typedef std::map<std::string, std::shared_ptr<std::vector<float>>> tensor_map_t;
typedef std::map<std::string, std::vector<int64_t>> shape_map_t;

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

class PyCallBack : public CallBack {
public:
  explicit PyCallBack(py::function &func) : run_(func) {}
  void run(std::string layer_name) { run_(layer_name); }
  py::function run_;
};

// Warning: buffer in C++. New inference will erase old output
static py::array getPyArray(std::shared_ptr<std::vector<float>> ptr,
                            const std::vector<int64_t> &shape) {
  auto shared_ptr_ptr = new std::shared_ptr<std::vector<float>>(std::move(ptr));
  py::capsule delete_shared_ptr_ptr(shared_ptr_ptr, [](void *ptr) {
    delete reinterpret_cast<std::shared_ptr<std::vector<float>> *>(ptr);
  });
  return py::array_t<float>(shape, (*shared_ptr_ptr)->data(),
                            delete_shared_ptr_ptr);
}

struct quant_brief_info {
  std::string dtype;
  std::string shape;
  float scale;
  int zp;
};

class py_module {
public:
  py_module() {}
  ~py_module() {
    interpreter_.reset();
    auto module = module_.release();
    if (module) {
      module.erase();
    }
    context_.reset();
  }
  void load(std::string filename) {
    if (context_) {
      context_.reset();
    }

    DialectRegistry registry;
    registry.insert<func::FuncDialect, top::TopDialect, tpu::TpuDialect,
                    quant::QuantizationDialect>();
    context_ = std::make_unique<MLIRContext>(registry);

    module_ = parseSourceFile<ModuleOp>(filename, context_.get());
    assert(module_);
    if (interpreter_) {
      interpreter_.reset();
    }

    interpreter_ = std::make_unique<ModuleInterpreter>(module_.get());
    interpreter_->set_mem_mode(gmem_mode_str_);
    interpreter_->allocate_resources();
    for (auto &name : interpreter_->input_names) {
      input_names.append(name);
    }
    for (auto &name : interpreter_->output_names) {
      output_names.append(name);
    }
    for (auto &name : interpreter_->all_tensor_names) {
      all_tensor_names.append(name);
    }
    for (auto &name : interpreter_->all_weight_names) {
      all_weight_names.append(name);
    }
  }

  py::dict getAllTensor() {
    py::dict py_ret;
    for (auto &name : interpreter_->all_tensor_names) {
      if (!interpreter_->hasTensorMem(name)) {
        // skip when part mem or memory allocated failed.
        continue;
      }
      auto tensor = interpreter_->getTensor(name);
      auto shape = interpreter_->getTensorShape(name);
      py::str py_s(name);
      py_ret[py_s] = getPyArray(std::move(tensor), shape);
    }
    return py_ret;
  }

  void before_invoke(py::function func) {
    auto py_ptr = std::make_shared<PyCallBack>(func);

    interpreter_->before_hooks.push_back(std::move(py_ptr));
  }

  void after_invoke(py::function func) {
    auto py_ptr = std::make_shared<PyCallBack>(func);
    interpreter_->after_hooks.push_back(std::move(py_ptr));
  }

  void clear_hooks() { interpreter_->clear_hooks(); }

  static void set_mem_mode(std::string mem_mode) {
    py_module::gmem_mode_str_ = mem_mode;
  }
  void set_tensor(
      std::string name,
      py::array_t<float, py::array::c_style | py::array::forcecast> data) {
    interpreter_->setTensor(name, data.data(), data.size() * sizeof(float),
                            false);
  }

  void set_tensor_from_int(
      std::string name,
      py::array_t<float, py::array::c_style | py::array::forcecast> data) {
    interpreter_->setTensor(name, data.data(), data.size() * sizeof(float),
                            true);
  }

  // Warning: using copy in python
  py::array get_tensor(std::string name) {
    auto tensor = interpreter_->getTensor(name);
    auto shape = interpreter_->getTensorShape(name);
    return getPyArray(std::move(tensor), shape);
  }

  // Tip: not using copy in python, since independent mem
  py::array get_fp32_tensor(std::string name) {
    auto tensor = interpreter_->getTensor(name, true);
    auto shape = interpreter_->getTensorShape(name);
    return getPyArray(std::move(tensor), shape);
  }

  struct quant_brief_info format_tensor_qinfo(std::string name) {
    struct quant_brief_info q_info;
    if (!interpreter_->getTensorQuantInfo(name, q_info.dtype, q_info.scale,
                                          q_info.zp)) {
      q_info.dtype = std::string("NA");
      q_info.scale = 1.0;
      q_info.zp = 0;
      q_info.shape = std::string("[]");
      return q_info;
    }
    std::vector<int64_t> shape_ = interpreter_->getTensorShape(name);
    q_info.shape = std::string("[");
    for (int i = 0; i < shape_.size(); i++) {
      q_info.shape += std::to_string(shape_[i]);
      if (i != shape_.size() - 1)
        q_info.shape += std::string(", ");
    }
    q_info.shape += std::string("]");
    return q_info;
  }

  void invoke(bool fixed_to_float) { interpreter_->invoke(fixed_to_float); }
  void fake_quant_weight() { interpreter_->fake_quant_weight(); }

  py::array invoke_at(const std::string name) {
    auto tensor = interpreter_->invoke_at(name);
    auto shape = interpreter_->getTensorShape(name);
    return getPyArray(std::move(tensor), shape);
  }

  py::array backward_weight_at(
      const std::string name, const std::string weight_name,
      py::array_t<float, py::array::c_style | py::array::forcecast> grd_dst) {
    auto shape = interpreter_->getTensorShape(weight_name);
    size_t size = 1;
    for (auto dim : shape) {
      size *= dim;
    }
    py::array_t<float, py::array::c_style | py::array::forcecast> weight_grd(
        shape);
    interpreter_->backward_weight_at(name, grd_dst.data(), grd_dst.size(),
                                     weight_grd.data(), size);
    return weight_grd;
  }

  void invoke_from(const std::string name) { interpreter_->invoke_from(name); }

public:
  py::list all_tensor_names;
  py::list all_weight_names;
  py::list input_names;
  py::list output_names;
  static std::string version;
  static std::string gmem_mode_str_;

private:
  std::unique_ptr<mlir::MLIRContext> context_;
  OwningOpRef<ModuleOp> module_;
  std::string weightFilePath_;
  std::unique_ptr<ModuleInterpreter> interpreter_;
};

void set_mem_mode(std::string mem_mode) { py_module::set_mem_mode(mem_mode); }

void debug_only(std::vector<std::string> debug_types) {
  llvm::DebugFlag = true;
  std::vector<const char *> c_debug;
  c_debug.reserve(debug_types.size());

  for (auto &d : debug_types)
    c_debug.push_back(const_cast<char *>(d.c_str()));
  llvm::setCurrentDebugTypes(c_debug.data(), c_debug.size());
}

void debug(bool enable) { llvm::DebugFlag = enable; }

#ifndef MLIR_VERSION
#define MLIR_VERSION "version unknown"
#endif

std::string py_module::version = MLIR_VERSION;
std::string py_module::gmem_mode_str_ = "";


void run_pass_pipeline(std::string mlir_txt, std::vector<std::string> opts) {
  tpu_mlir::registerAllPasses();

  DialectRegistry registry;
  tpu_mlir::registerAllDialects(registry);
  MLIRContext context(registry);

  for (auto name : registry.getDialectNames())
    context.getOrLoadDialect(name);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(mlir_txt, &context);
  auto pm = PassManager::on<ModuleOp>(&context);
  auto errorHandler = [&](const Twine &msg) {
    emitError(UnknownLoc::get(pm.getContext())) << msg;
    return failure();
  };
  for (auto opt: opts) {
    auto n = opt.size();
    if (opt[0] == '-' && opt[1] == '-' && n > 2) {
      int i = opt.find_first_of("=");
      if (i == -1) {
        i = n;
      } else if (opt[i + 1] != '\"' || opt[n - 1] != '\"') {
        continue;
      }
      auto pass_name = opt.substr(2, i - 2);
      const PassInfo *passInfo = Pass::lookupPassInfo(pass_name);
      if (!passInfo) {
        llvm::errs() << "unknown pass: " << pass_name << "\n";
        continue;
      } else {
        auto pass_option = (i == n) ? "" : opt.substr(i + 2, n - i - 3);
        passInfo->addToPipeline(pm, pass_option, errorHandler);
      }
    }
  }

  pm.run(*module);
}

// wrap as Python module
PYBIND11_MODULE(pymlir, m) {
  m.doc() = "pybind11 for mlir";
  m.def("debug", &debug, py::arg("enable") = true,
        "enable debugging information");
  m.def("debug", &debug_only, "configure debugging information");
  m.def("set_mem_mode", &set_mem_mode, "set_Gmem_mode_str");
  m.def("run_pass_pipeline", &run_pass_pipeline, "run_pass_pipeline");
  py::class_<quant_brief_info>(m, "q_info", "simple tensor quant info")
      .def_readwrite("dtype", &quant_brief_info::dtype)
      .def_readwrite("shape", &quant_brief_info::shape)
      .def_readwrite("scale", &quant_brief_info::scale)
      .def_readwrite("zp", &quant_brief_info::zp);
  // clang-format off
  py::class_<py_module>(m, "module", "MLIR Module")
      .def(py::init<>())
      .def("load", &py_module::load, "load module from IR")
      .def("set_mem_mode", &py_module::set_mem_mode)
      .def("set_tensor", &py_module::set_tensor)
      .def("set_tensor_from_int", &py_module::set_tensor_from_int)
      .def("get_tensor", &py_module::get_tensor, "get one tensor data")
      .def("get_fp32_tensor", &py_module::get_fp32_tensor, "get one fp32 tensor data")
      .def("get_all_tensor", &py_module::getAllTensor, "dump all tensor data")
      .def("invoke", &py_module::invoke, py::arg("fixed_to_float")=true)
      .def("fake_quant_weight", &py_module::fake_quant_weight)
      .def("invoke_at", &py_module::invoke_at, "invote at specified layer")
      .def("backward_weight_at", &py_module::backward_weight_at, "invoke the backward weight function of conv op")
      .def("invoke_from", &py_module::invoke_from, "invote from specified layer to the end")
      .def("get_tensor_qinfo", &py_module::format_tensor_qinfo, "get simple quant info of tensor")
      .def("before_invoke", &py_module::before_invoke, "add a before hook")
      .def("after_invoke", &py_module::after_invoke, "add a before hook")
      .def("clear_hooks", &py_module::clear_hooks, "clear hooks")
      .def_readonly("input_names", &py_module::input_names)
      .def_readonly("output_names", &py_module::output_names)
      .def_readonly("all_tensor_names", &py_module::all_tensor_names)
      .def_readonly("all_weight_names", &py_module::all_weight_names)
      .def_readonly_static("version", &py_module::version);

  py::scoped_ostream_redirect output{std::cerr, py::module::import("sys").attr("stderr")};
}

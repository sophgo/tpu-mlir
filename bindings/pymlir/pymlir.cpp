//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

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
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
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

template <typename Dtype>
static py::array getPythonArray(std::vector<Dtype> *vec,
                                const std::vector<int64_t> &shape) {
  std::vector<unsigned> stride_v(shape.size(), sizeof(Dtype));
  for (int i = shape.size() - 1; i > 0; i--) {
    for (int j = 0; j < i; j++) {
      stride_v[j] *= shape[i];
    }
  }

  return py::array(
      py::buffer_info(vec->data(),   /* data as contiguous array  */
                      sizeof(Dtype), /* size of one scalar        */
                      py::format_descriptor<Dtype>::format(), /* data type */
                      shape.size(),                           // ndim/
                      shape,                                  // shape
                      stride_v                                // strides
                      ));
}

static py::dict getTensorDict(tensor_map_t &tensorMap, shape_map_t &shapeMap,
                              std::vector<std::string> &ordered_names) {
  py::dict py_ret;
  for (auto &name : ordered_names) {
    py::str py_s(name);

    assert(shapeMap.end() != shapeMap.find(name));
    py_ret[py_s] =
        getPythonArray(tensorMap.find(name)->second.get(), shapeMap[name]);
  }

  return py_ret;
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
  }

  py::dict getAllTensor() {
    tensor_map_t tensorMap_;
    shape_map_t shapeMap_;
    std::vector<std::string> ordered_names;
    auto &all_tensor_names = interpreter_->all_tensor_names;
    for (auto &tensor_name : all_tensor_names) {
      ordered_names.emplace_back(tensor_name);
      tensorMap_[tensor_name] = interpreter_->getTensor(tensor_name);
      shapeMap_[tensor_name] = interpreter_->getTensorShape(tensor_name);
    }

    return getTensorDict(tensorMap_, shapeMap_, ordered_names);
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

  py::array get_tensor(std::string name) {
    auto tensor = interpreter_->getTensor(name);
    std::vector<int64_t> shape = interpreter_->getTensorShape(name);
    return getPythonArray(tensor.get(), shape);
  }

  py::array get_fp32_tensor(std::string name) {
    auto tensor = interpreter_->getTensor(name, true);
    std::vector<int64_t> shape = interpreter_->getTensorShape(name);
    return getPythonArray(tensor.get(), shape);
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

  void invoke() { interpreter_->invoke(); }
  void fake_quant_weight() { interpreter_->fake_quant_weight(); }

  py::array invoke_at(const std::string name) {
    auto tensor = interpreter_->invoke_at(name);
    std::vector<int64_t> shape = interpreter_->getTensorShape(name);
    return getPythonArray(tensor.get(), shape);
  }

  void invoke_from(const std::string name) { interpreter_->invoke_from(name); }

public:
  py::list all_tensor_names;
  py::list input_names;
  py::list output_names;
  static std::string version;

private:
  std::unique_ptr<mlir::MLIRContext> context_;
  OwningOpRef<ModuleOp> module_;
  std::string weightFilePath_;
  std::unique_ptr<ModuleInterpreter> interpreter_;
};

void debug(std::vector<std::string> debug_types) {
  llvm::DebugFlag = true;
  std::vector<const char *> c_debug;
  c_debug.reserve(debug_types.size());
  for (auto &d : debug_types)
    c_debug.push_back(const_cast<char *>(d.c_str()));
  llvm::setCurrentDebugTypes(c_debug.data(), c_debug.size());
}

#ifndef MLIR_VERSION
#define MLIR_VERSION "version unknown"
#endif

std::string py_module::version = MLIR_VERSION;

// wrap as Python module
PYBIND11_MODULE(pymlir, m) {
  m.doc() = "pybind11 for mlir";
  m.def("debug", &debug, "configure debugging information");

  py::class_<quant_brief_info>(m, "q_info", "simple tensor quant info")
      .def_readwrite("dtype", &quant_brief_info::dtype)
      .def_readwrite("shape", &quant_brief_info::shape)
      .def_readwrite("scale", &quant_brief_info::scale)
      .def_readwrite("zp", &quant_brief_info::zp);

  py::class_<py_module>(m, "module", "MLIR Module")
      .def(py::init<>())
      .def("load", &py_module::load, "load module from IR")
      .def("set_tensor", &py_module::set_tensor)
      .def("set_tensor_from_int", &py_module::set_tensor_from_int)
      .def("get_tensor", &py_module::get_tensor, "get one tensor data")
      .def("get_fp32_tensor", &py_module::get_fp32_tensor, "get one fp32 tensor data")
      .def("get_all_tensor", &py_module::getAllTensor, "dump all tensor data")
      .def("invoke", &py_module::invoke)
      .def("fake_quant_weight", &py_module::fake_quant_weight)
      .def("invoke_at", &py_module::invoke_at, "invote at specified layer")
      .def("invoke_from", &py_module::invoke_from, "invote from specified layer to the end")
      .def("get_tensor_qinfo", &py_module::format_tensor_qinfo, "get simple quant info of tensor")
      .def_readonly("input_names", &py_module::input_names)
      .def_readonly("output_names", &py_module::output_names)
      .def_readonly("all_tensor_names", &py_module::all_tensor_names)
      .def_readonly_static("version", &py_module::version);
}

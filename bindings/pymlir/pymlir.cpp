//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "pymodule.h"

#ifdef USE_CUDA
#include "pycuda.h"
#endif

#ifndef MLIR_VERSION
#define MLIR_VERSION "version unknown"
#endif
static std::string mlir_version = MLIR_VERSION;

#ifdef USE_CUDA
static bool support_cuda = true;
#else
static bool support_cuda = false;
#endif

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

void run_pass_pipeline(std::string mlir_txt, std::vector<std::string> opts) {
  tpu_mlir::registerAllPasses();

  DialectRegistry registry;
  tpu_mlir::registerAllDialects(registry);
  MLIRContext context(registry);

  for (auto name : registry.getDialectNames())
    context.getOrLoadDialect(name);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(mlir_txt, &context);
  auto pm = PassManager::on<ModuleOp>(&context);
  auto errorHandler = [&](const Twine &msg) {
    emitError(UnknownLoc::get(pm.getContext())) << msg;
    return failure();
  };
  for (auto opt : opts) {
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
  m.doc() = "pybind11 for TPU-MLIR";
  m.attr("__version__") = mlir_version;
  m.attr("support_cuda") = support_cuda;
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
      .def_readonly("all_weight_names", &py_module::all_weight_names);
#ifdef USE_CUDA
  py::class_<py_cuda>(m, "cuda", "MLIR Cuda")
      .def(py::init<>())
      .def("load", &py_cuda::load, "load mlir to cuda")
      .def("set_tensor", &py_cuda::set_tensor)
      .def("invoke", &py_cuda::invoke, py::arg("dump_all")=false)
      .def("get_tensor", &py_cuda::get_tensor, "get one tensor data")
      .def("get_all_tensor", &py_cuda::get_all_tensor, "dump all tensor data")
      .def_readonly("input_names", &py_cuda::input_names)
      .def_readonly("output_names", &py_cuda::output_names);
#endif
  // clang-format on
  py::scoped_ostream_redirect output{std::cerr,
                                     py::module::import("sys").attr("stderr")};
}

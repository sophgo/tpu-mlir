//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "tpu_mlir/InitAll.h"
#include "pyfinalmlir.h"

py_final_module::~py_final_module() {
  evaluator_.reset();
  auto module = module_.release();
  if (module) {
    module.erase();
  }
  context_.reset();
}

void py_final_module::load(std::string filename) {
  if (context_) {
    context_.reset();
  }

  DialectRegistry registry;
  registry.insert<func::FuncDialect, top::TopDialect, tpu::TpuDialect,
                  quant::QuantizationDialect>();
  context_ = std::make_unique<MLIRContext>(registry);

  module_ = parseSourceFile<ModuleOp>(filename, context_.get());
  assert(module_);
  if (evaluator_) {
    evaluator_.reset();
  }

  evaluator_ = std::make_unique<tpu::BM168xEvaluator>(module_.get());
  evaluator_->allocate_resources();
  for (auto &name : evaluator_->input_names) {
    input_names.append(name);
  }
  for (auto &name : evaluator_->output_names) {
    output_names.append(name);
  }
  for (auto &name : evaluator_->all_tensor_names) {
    all_tensor_names.append(name);
  }
  for (auto &name : evaluator_->all_weight_names) {
    all_weight_names.append(name);
  }
}

py::dict py_final_module::getAllTensor() {
  py::dict py_ret;
  for (auto &name : evaluator_->all_tensor_names) {
    auto tensor = evaluator_->getTensor(name);
    auto shape = evaluator_->getTensorShape(name);
    py::str py_s(name);
    py_ret[py_s] = getPyArray(std::move(tensor), shape);
  }
  return py_ret;
}

void py_final_module::set_tensor(
    std::string name,
    py::array_t<float, py::array::c_style | py::array::forcecast> data) {
  evaluator_->setTensor(name, data.data(), data.size() * sizeof(float), false);
}

void py_final_module::set_tensor_from_int(
    std::string name, py::array_t<float, py::array::c_style | py::array::forcecast> data) {
  evaluator_->setTensor(name, data.data(), data.size() * sizeof(float), true);
}

void py_final_module::invoke() {
  evaluator_->invoke();
}

PYBIND11_MODULE(pyfinalmlir, m) {
  py::class_<py_final_module>(m, "module", "MLIR Final Module")
      .def(py::init<>())
      .def("load", &py_final_module::load, "load module from IR")
      .def("set_tensor", &py_final_module::set_tensor)
      .def("set_tensor_from_int", &py_final_module::set_tensor_from_int)
      .def("get_all_tensor", &py_final_module::getAllTensor, "dump all tensor data")
      .def("invoke", &py_final_module::invoke)
      .def_readonly("input_names", &py_final_module::input_names)
      .def_readonly("output_names", &py_final_module::output_names)
      .def_readonly("all_tensor_names", &py_final_module::all_tensor_names)
      .def_readonly("all_weight_names", &py_final_module::all_weight_names);
  // clang-format on
  py::scoped_ostream_redirect output{std::cerr,
                                     py::module::import("sys").attr("stderr")};
}

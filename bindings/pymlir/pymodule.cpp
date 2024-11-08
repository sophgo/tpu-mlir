//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "pymodule.h"

std::string py_module::gmem_mode_str_ = "";

py_module::~py_module() {
  interpreter_.reset();
  auto module = module_.release();
  if (module) {
    module.erase();
  }
  context_.reset();
}

void py_module::load(std::string filename) {
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

py::dict py_module::getAllTensor() {
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

void py_module::before_invoke(py::function func) {
  auto py_ptr = std::make_shared<PyCallBack>(func);

  interpreter_->before_hooks.push_back(std::move(py_ptr));
}

void py_module::after_invoke(py::function func) {
  auto py_ptr = std::make_shared<PyCallBack>(func);
  interpreter_->after_hooks.push_back(std::move(py_ptr));
}

void py_module::clear_hooks() { interpreter_->clear_hooks(); }

void py_module::set_mem_mode(std::string mem_mode) {
  py_module::gmem_mode_str_ = mem_mode;
}

void py_module::set_tensor(
    std::string name,
    py::array_t<float, py::array::c_style | py::array::forcecast> data,
    std::vector<int64_t> shape) {
  interpreter_->setTensor(name, data.data(), data.size(), shape,
                          false);
}

void py_module::set_tensor_from_int(
    std::string name,
    py::array_t<float, py::array::c_style | py::array::forcecast> data,
    std::vector<int64_t> shape) {
  interpreter_->setTensor(name, data.data(), data.size(), shape, true);
}

// Warning: using copy in python
py::array py_module::get_tensor(std::string name) {
  auto tensor = interpreter_->getTensor(name);
  auto shape = interpreter_->getTensorShape(name);
  return getPyArray(std::move(tensor), shape);
}

// Tip: not using copy in python, since independent mem
py::array py_module::get_fp32_tensor(std::string name) {
  auto tensor = interpreter_->getTensor(name, true);
  auto shape = interpreter_->getTensorShape(name);
  return getPyArray(std::move(tensor), shape);
}

struct quant_brief_info py_module::format_tensor_qinfo(std::string name) {
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

void py_module::invoke(bool fixed_to_float) {
  interpreter_->invoke(fixed_to_float);
}
void py_module::fake_quant_weight() { interpreter_->fake_quant_weight(); }

py::array py_module::invoke_at(const std::string name) {
  auto tensor = interpreter_->invoke_at(name);
  auto shape = interpreter_->getTensorShape(name);
  return getPyArray(std::move(tensor), shape);
}

py::array py_module::backward_weight_at(
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

void py_module::invoke_from(const std::string name) {
  interpreter_->invoke_from(name);
}

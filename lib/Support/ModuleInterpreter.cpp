//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/ModuleInterpreter.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include <llvm/Support/Debug.h>
#include "progressbar.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>

#define DEBUG_TYPE "interpreter"

namespace tpu_mlir {
ModuleInterpreter::ModuleInterpreter(ModuleOp module) : module(module) {
  module::init(module);
  if (!module::isState(module::State::TOP_F32) &&
      !module::isState(module::State::TPU_LOWERED)) {
    llvm_unreachable("mlir state not support");
  }
}

ModuleInterpreter::~ModuleInterpreter() {
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (auto infer_op = llvm::dyn_cast<InferenceInterface>(op)) {
        auto name = module::getName(op).str();
        infer_op.deinit(*inference_map[name]);
      }
    });
  }
}

bool ModuleInterpreter::is_no_mem_op(Operation *op) {
  if (op == nullptr) {
    return false;
  }
  return isa<top::ReshapeOp, tpu::ReshapeOp, top::UnsqueezeOp, top::SqueezeOp>(
      op);
}

void ModuleInterpreter::allocate_resources() {
  all_tensor_names.clear();
  value_map.clear();
  mem_map.clear();
  num_infer_op = 0;
  int64_t total_count = 0;
  for (auto func : module.getOps<FuncOp>()) {
    // if (func.getName() != "main") {
    //   continue;
    // }
    // alloce buffer for all value
    func.walk([&](Operation *op) {
      if (op == func.getOperation() || isa<top::NoneOp>(op)) {
        // self
      } else if (isa<ReturnOp>(op)) {
        for (auto v : op->getOperands()) {
          auto name = module::getName(v).str();
          output_names.push_back(name);
        }
      } else {
        for (auto result : op->getResults()) {
          auto type = result.getType().dyn_cast<RankedTensorType>();
          if (!type) {
            continue;
          }
          auto count = type.getNumElements();
          total_count += count;
          auto name = module::getName(result).str();
          value_map[name] = result;
          if (auto wOp = llvm::dyn_cast<top::WeightOp>(op)) {
            mem_map[name] = wOp.read_as_float();
            all_weight_names.push_back(name);
          } else if (is_no_mem_op(op)) {
            // output with the same memory of input
            auto in = module::getName(op->getOperand(0));
            mem_map[name] = mem_map[in.str()];
            all_tensor_names.push_back(name);
          } else {
            mem_map[name] = std::make_shared<std::vector<float>>(count);
            all_tensor_names.push_back(name);
          }
          if (isa<top::InputOp>(op)) {
            input_names.push_back(name);
          }
        }
      }
    });
    module::detachWeightFile(); // to free weight memory
    for (auto &name : output_names) {
      if (std::find(all_tensor_names.begin(), all_tensor_names.end(), name) ==
          all_tensor_names.end()) {
        // if weight is output, then dump it
        all_tensor_names.push_back(name);
      }
    }

    // input output buffers for all ops
    func.walk([&](Operation *op) {
      if (auto infer_op = llvm::dyn_cast<InferenceInterface>(op)) {
        num_infer_op++;
        auto name = module::getName(op).str();
        auto param = std::make_shared<InferenceParameter>();
        for (auto result : op->getResults()) {
          if (result.getType().isa<NoneType>()) {
            param->outputs.push_back(nullptr);
          } else {
            auto o_name = module::getName(result).str();
            param->outputs.push_back(mem_map[o_name]->data());
          }
        }
        for (auto input : op->getOperands()) {
          if (input.getType().isa<NoneType>()) {
            param->inputs.push_back(nullptr);
            continue;
          }
          auto input_name = module::getName(input).str();
          if (mem_map.find(input_name) == mem_map.end()) {
            input.dump();
            llvm_unreachable("input operands not allocated");
          } else {
            param->inputs.push_back(mem_map[input_name]->data());
          }
        }
        LLVM_DEBUG(llvm::dbgs() << "init: '" << name << "'\n");
        if (failed(infer_op.init(*param))) {
          op->dump();
          llvm_unreachable("op inferece init failed");
        }
        inference_map[name] = param;
      }
    });
  }
  LLVM_DEBUG(llvm::dbgs() << "Allocate size: " << total_count * 4 / 1024
                          << " KB\n");
}

void ModuleInterpreter::fake_quant_weight() {
  module::init(module);
  LLVM_DEBUG(llvm::errs() << "start fake_quant_weight\n");
  std::vector<std::string> not_quant_weight_names;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (isa<top::ConvOp>(op) || isa<top::MatMulOp>(op)) {
        auto bias_op = op->getOperands()[2].getDefiningOp();
        if (auto weight_op = dyn_cast<top::WeightOp>(bias_op)) {
          not_quant_weight_names.push_back(module::getName(bias_op).str());
        }
      }
    });
  }

  for (auto &name : all_weight_names) {
    if (std::count(not_quant_weight_names.begin(), not_quant_weight_names.end(),
                   name)) {
      continue;
    }

    auto mem = *mem_map.at(name);
    auto max_value =
        std::max(std::abs(*std::max_element(mem.begin(), mem.end())),
                 std::abs(*std::min_element(mem.begin(), mem.end())));
    for (auto &data : mem) {
      data = std::round(data * 127 / max_value) * max_value / 127;
    }
  }
}

void ModuleInterpreter::invoke(bool express_type) {
  module::init(module);
  progressbar bar(num_infer_op);
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](InferenceInterface infer_op) {
      bar.update();
      auto name = module::getName(infer_op.getOperation()).str();
      LLVM_DEBUG(llvm::dbgs() << "compute: '" << infer_op << "'\n");
      if (failed(infer_op.inference(*inference_map[name]))) {
        infer_op.dump();
        llvm_unreachable("invoke failed!!");
      }
    });
  }
  llvm::errs() << "\n";
  if (express_type && module::isState(module::State::TPU_LOWERED)) {
    for (auto &name : all_tensor_names) {
      auto value = value_map.at(name);
      if (is_no_mem_op(value.getDefiningOp())) {
        continue;
      }
      auto mem = mem_map.at(name);
      if (module::isUniformQuantized(value)) {
        auto qtype = module::getUniformQuantizedType(value);
        for (auto &data : *mem) {
          data = (data - (float)qtype.getZeroPoint()) * (float)qtype.getScale();
        }
      }
    }
  }
}

std::shared_ptr<std::vector<float>>
ModuleInterpreter::invoke_at(const std::string op_name) {
  module::init(module);
  if (value_map.find(op_name) == value_map.end()) {
    llvm::errs() << "Can't find op:" << op_name << "\n";
    llvm_unreachable("invoke_at op_name error");
  }
  auto v = value_map[op_name];
  auto op = v.getDefiningOp();
  if (op == nullptr || false == isa<InferenceInterface>(op)) {
    llvm::errs() << "Op :" << op_name << " can't do inference";
    llvm_unreachable("invoke_at infer error");
  }
  auto infer_op = cast<InferenceInterface>(op);
  LLVM_DEBUG(llvm::dbgs() << "invoke at: '" << infer_op << "'\n");
  if (failed(infer_op.inference(*inference_map[op_name]))) {
    infer_op.dump();
    llvm_unreachable("infer_op.inference failed!!");
  }

  return getTensor(op_name);
}

void ModuleInterpreter::invoke_from(const std::string op_name) {
  module::init(module);
  bool start_run = false;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](InferenceInterface infer_op) {
      auto name = module::getName(infer_op).str();
      if (name == op_name) {
        start_run = true;
      }
      LLVM_DEBUG(llvm::dbgs() << "invoke: '" << infer_op << "'\n");
      if (start_run && failed(infer_op.inference(*inference_map[name]))) {
        infer_op.dump();
        llvm_unreachable("invoke failed!!");
      }
    });
  }
}
void ModuleInterpreter::setTensor(const std::string &name, const void *data,
                                  size_t size, bool is_integer) {
  module::init(module);
  auto it = mem_map.find(name);
  if (it == mem_map.end()) {
    llvm::errs() << "Can't find op name: " << name << "\n";
    llvm_unreachable("Error, setTensor failed");
  }
  auto act = it->second;
  if (act->size() * sizeof(float) != size) {
    llvm::errs() << "Tensor " << name
                 << " data need size: " << act->size() * sizeof(float)
                 << " , but set size: " << size << "\n";
    llvm_unreachable("Error, setTensor failed");
  }
  auto value = value_map.at(name);
  if (is_integer == false && module::isUniformQuantized(value)) {
    auto qtype = module::getUniformQuantizedType(value);
    float *p = (float *)data;
    for (uint32_t i = 0; i < act->size(); i++) {
      float d =
          p[i] * (float)(1 / qtype.getScale()) + (float)qtype.getZeroPoint();
      act->at(i) = qtype.isSigned() ? to_int8(d) : to_uint8(d);
    }
  } else {
    memcpy(act->data(), data, size);
  }
}

std::shared_ptr<std::vector<float>>
ModuleInterpreter::getTensor(const std::string &name, bool express_type) {
  auto it = mem_map.find(name);
  if (it == mem_map.end()) {
    llvm::errs() << "Can't find op name: " << name << "\n";
    llvm_unreachable("Error, getTensor failed");
  }

  if (express_type && module::isState(module::State::TPU_LOWERED)) {
    auto value = value_map.at(name);
    if (module::isUniformQuantized(value)) {
      int i = 0;
      auto mem = mem_map.at(name);
      auto data_fp32 = std::make_shared<std::vector<float>>(it->second->size());
      auto qtype = module::getUniformQuantizedType(value);
      for (auto &data : *mem) {
        data_fp32->data()[i++] =
            (data - (float)qtype.getZeroPoint()) * (float)qtype.getScale();
      }
      return std::move(data_fp32);
    }
  }

  std::shared_ptr<std::vector<float>> tmp(it->second);
  return std::move(tmp);
}

bool ModuleInterpreter::getTensorQuantInfo(const std::string name,
                                           std::string &dtype, float &scale,
                                           int &zp) {
  auto it = mem_map.find(name);
  if (it == mem_map.end()) {
    return false;
  }
  auto value = value_map.at(name);
  auto stype = module::getStorageType(value);
  if (module::isUniformQuantized(value)) {
    auto qtype = module::getUniformQuantizedType(value);
    scale = qtype.getScale();
    zp = qtype.getZeroPoint();
    if (stype.isSignlessInteger(8) || stype.isUnsignedInteger(8))
      dtype = std::string("U8");
    else if (stype.isSignedInteger(8))
      dtype = std::string("I8");
    else if (stype.isSignlessInteger(16) || stype.isUnsignedInteger(16))
      dtype = std::string("U16");
    else if (stype.isSignedInteger(16))
      dtype = std::string("I16");
    else if (stype.isSignedInteger(32))
      dtype = std::string("I32");
    else if (stype.isSignlessInteger(32) || stype.isUnsignedInteger(32))
      dtype = std::string("U32");
    else {
      dtype = std::string("I4");
    }
  } else if (stype.isa<FloatType>()) {
    if (stype.isF16()) {
      dtype = std::string("F16");
      scale = 1.0;
      zp = 0;
    } else if (stype.isBF16()) {
      dtype = std::string("BF16");
      scale = 1.0;
      zp = 0;
    } else if (stype.isF32()) {
      dtype = std::string("F32");
      scale = 1.0;
      zp = 0;
    } else {
      dtype = std::string("NA");
      scale = 1.0;
      zp = 0;
    }
  }
  return true;
}

llvm::ArrayRef<int64_t>
ModuleInterpreter::getTensorShape(const std::string &name) {
  auto it = value_map.find(name);
  if (it == value_map.end()) {
    llvm::errs() << "Can't find op name: " << name << "\n";
    llvm_unreachable("Error, getTensorShape failed");
  }
  return it->second.getType().cast<RankedTensorType>().getShape();
}

} // namespace tpu_mlir

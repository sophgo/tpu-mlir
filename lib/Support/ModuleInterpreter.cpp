//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/ModuleInterpreter.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>

using namespace mlir;
using namespace mlir::func;
using namespace tpu_mlir::helper;

namespace tpu_mlir {
ModuleInterpreter::ModuleInterpreter(ModuleOp module) : module(module) {
  state = Module::getState(module);
  if (state != Module::State::TOP_F32 &&
      state != Module::State::TPU_LOWERED) {
    llvm_unreachable("mlir state not support");
  }
}

ModuleInterpreter::~ModuleInterpreter() {
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (auto infer_op = llvm::dyn_cast<InferenceInterface>(op)) {
        auto name = op->getAttrOfType<StringAttr>("name").str();
        infer_op.deinit(*inference_map[name]);
      }
    });
  }
}

void ModuleInterpreter::allocate_resources() {
  all_tensor_names.clear();
  value_map.clear();
  mem_map.clear();
  for (auto func : module.getOps<FuncOp>()) {
    // if (func.getName() != "main") {
    //   continue;
    // }
    // alloce buffer for all value
    func.walk([&](Operation *op) {
      if (op == func.getOperation() || isa<top::NoneOp>(op)) {
        // self
      } else if (isa<func::ReturnOp>(op)) {
        for (auto v : op->getOperands()) {
          auto opd = v.getDefiningOp();
          auto name = opd->getAttrOfType<StringAttr>("name").str();
          output_names.push_back(name);
        }
      } else {
        auto result = op->getResult(0);
        auto type = result.getType().cast<RankedTensorType>();
        auto count = type.getNumElements();
        auto name = op->getAttrOfType<StringAttr>("name").str();
        value_map[name] = result;
        if (auto wOp = llvm::dyn_cast<top::WeightOp>(op)) {
          mem_map[name] = wOp.read_as_float();
        } else {
          mem_map[name] = std::make_shared<std::vector<float>>(count);
          all_tensor_names.push_back(name);
        }
        if (isa<top::InputOp>(op)) {
          input_names.push_back(name);
        }
      }
    });

    llvm::errs() << "fill InferenceParameter\n";
    // input output buffers for all ops
    func.walk([&](Operation *op) {
      if (auto infer_op = llvm::dyn_cast<InferenceInterface>(op)) {
        auto name = op->getAttrOfType<StringAttr>("name").str();
        auto param = std::make_shared<InferenceParameter>();
        param->outputs.push_back(mem_map[name]->data());
        for (auto input : op->getOperands()) {
          if (input.getType().isa<NoneType>()) {
            param->inputs.push_back(nullptr);
            continue;
          }
          auto input_name =
              input.getDefiningOp()->getAttrOfType<StringAttr>("name").str();
          if (mem_map.find(input_name) == mem_map.end()) {
            input.dump();
            llvm_unreachable("input operands not allocated");
          } else {
            param->inputs.push_back(mem_map[input_name]->data());
          }
        }
        if (failed(infer_op.init(*param))) {
          op->dump();
          llvm_unreachable("op inferece init failed");
        }
        inference_map[name] = param;
      }
    });
  }
}

void ModuleInterpreter::invoke(bool express_type) {
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](InferenceInterface infer_op) {
      auto name = infer_op->getAttrOfType<StringAttr>("name").str();
      if (failed(infer_op.inference(*inference_map[name]))) {
        infer_op.dump();
        llvm_unreachable("invoke failed!!");
      }
    });
  }
  if (express_type && state == Module::State::TPU_LOWERED) {
    for (auto &name : all_tensor_names) {
      auto mem = mem_map.at(name);
      auto value = value_map.at(name);
      if (Quant::isUniformQuantized(value)) {
        auto qtype = Quant::getUniformQuantizedType(value);
        for (auto &data : *mem) {
          data = (data - qtype.getZeroPoint()) * qtype.getScale();
        }
      }
    }
  }
}

void ModuleInterpreter::setTensor(const std::string &name, const void *data,
                                  size_t size) {
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
  memcpy(act->data(), data, size);
}

std::shared_ptr<std::vector<float>>
ModuleInterpreter::getTensor(const std::string &name) {
  auto it = mem_map.find(name);
  if (it == mem_map.end()) {
    llvm::errs() << "Can't find op name: " << name << "\n";
    llvm_unreachable("Error, setTensor failed");
  }
  return std::move(it->second);
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

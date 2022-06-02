//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <sstream>
#include <fstream>
#include <set>

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {

static constexpr llvm::StringRef FUNC_TPU = "TPU";
static constexpr llvm::StringRef FUNC_CPU = "CPU";
static constexpr llvm::StringRef FUNC_SCF = "SCF"; // Structured Control Flow

class SubFunction {
public:
  SubFunction(StringRef mode) : mode(mode) {
    count++;
    have_none = false;
  }
  StringRef mode; // tpu/cpu/control
  std::vector<Operation *> ops;
  bool have_none;
  static int count;
};
int SubFunction::count = 0;

void getInputsOutputs(std::vector<Operation *> &ops, std::vector<Value> &inputs,
                      std::vector<Value> &outputs) {
  std::vector<Value> allValues;
  for (auto op : ops) {
    for (auto v : op->getResults()) {
      allValues.push_back(v);
    }
  }
  for (auto op : ops) {
    for (auto v : op->getOperands()) {
      if (find(inputs.begin(), inputs.end(), v) != inputs.end()) {
        continue;
      }
      auto inOp = v.getDefiningOp();
      if (isa<top::NoneOp>(inOp)) {
        continue;
      }
      if (find(allValues.begin(), allValues.end(), v) == allValues.end()) {
        inputs.push_back(v);
      }
    }
    auto v = op->getResult(0);
    if (find(outputs.begin(), outputs.end(), v) != outputs.end()) {
      continue;
    }
    for (auto use : v.getUsers()) {
      if (find(ops.begin(), ops.end(), use) == ops.end()) {
        outputs.push_back(v);
        break;
      }
    }
  }
}

void buildSubFunction(std::shared_ptr<SubFunction> sf, ModuleOp module) {
  // std::vector<Operation *> fnOps;
  std::vector<Value> fnInputs;
  std::vector<Value> fnOutputs;
  getInputsOutputs(sf->ops, fnInputs, fnOutputs);
  std::vector<Type> argType;
  std::vector<Type> resType;
  for (auto input : fnInputs) {
    argType.push_back(input.getType());
  }
  for (auto output : fnOutputs) {
    resType.push_back(output.getType());
  }
  int64_t id = SubFunction::count - 1;
  std::string func_name = "subfunc_" + std::to_string(id);
  OpBuilder builder(module.getContext());
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("id", builder.getI64IntegerAttr(id)));
  attrs.push_back(
      builder.getNamedAttr("mode", builder.getStringAttr(sf->mode)));
  auto fnType = builder.getFunctionType(llvm::ArrayRef<Type>{argType},
                                        llvm::ArrayRef<Type>{resType});
  auto fnOp = FuncOp::create(module.getLoc(), func_name, fnType,
                             ArrayRef<NamedAttribute>(attrs));
  auto block = fnOp.addEntryBlock();
  builder.setInsertionPointAfterValue(fnOutputs.back());
  func::CallOp callOp = builder.create<func::CallOp>(module.getLoc(), func_name,
                                                     resType, fnInputs);
  for (auto it : llvm::enumerate(callOp.getResults())) {
    fnOutputs[it.index()].replaceUsesWithIf(
        it.value(), [&](OpOperand &operand) {
          Operation *user = operand.getOwner();
          return find(sf->ops.begin(), sf->ops.end(), user) == sf->ops.end();
        });
  }
  builder.setInsertionPointToStart(block);
  top::NoneOp noneOp;
  if (sf->have_none) {
    noneOp =
        builder.create<top::NoneOp>(module.getLoc(), builder.getNoneType());
  }
  auto retOp = builder.create<func::ReturnOp>(module.getLoc(), fnOutputs);
  for (auto op : sf->ops) {
    if (isa<top::NoneOp>(op)) {
      continue;
    }
    for (auto it : llvm::enumerate(op->getOperands())) {
      if (isa<top::NoneOp>(it.value().getDefiningOp())) {
        op->setOperand(it.index(), noneOp);
      }
    }
    op->moveBefore(retOp);
  }
  module.push_back(fnOp);
  for (auto it : llvm::enumerate(fnInputs)) {
    auto arg = block->getArgument(it.index());
    it.value().replaceUsesWithIf(arg, [&](OpOperand &operand) {
      Operation *user = operand.getOwner();
      return find(sf->ops.begin(), sf->ops.end(), user) != sf->ops.end();
    });
  }
}

static StringRef getOpMode(Operation *op) {
  // if (isa<tpu::MaxPoolOp>(op)) {
  //   return FUNC_CPU; // here just simulate
  // }
  return FUNC_TPU;
}

static void insert_subop(std::shared_ptr<SubFunction> &subf, Operation *op) {
  for (auto opd : op->getOperands()) {
    auto op_ = opd.getDefiningOp();
    if (isa<top::WeightOp>(op_)) {
      subf->ops.push_back(op_);
    } else if (isa<top::NoneOp>(op_) && subf->have_none == false) {
      subf->have_none = true;
    }
  }
  subf->ops.push_back(op);
}

class SubnetDividePass : public SubnetDivideBase<SubnetDividePass> {
public:
  SubnetDividePass() {}
  void runOnOperation() override {
    auto module = getOperation();
    auto state = Module::getState(module);
    if (state != Module::State::TPU_REORDERED) {
      llvm_unreachable("module should be reordered");
    }
    auto mainFunc = Module::getMainFuncOp(module);
    std::shared_ptr<SubFunction> subf = nullptr;
    mainFunc.walk([&](Operation *op) {
      if (isa<top::InputOp, top::WeightOp, FuncOp, top::NoneOp, func::ReturnOp,
              func::CallOp>(op)) {
        // do nothing
      } else {
        auto mode = getOpMode(op);
        if (subf == nullptr) {
          subf = std::make_shared<SubFunction>(mode);
          insert_subop(subf, op);
        } else if (subf->mode == mode) {
          insert_subop(subf, op);
        } else {
          buildSubFunction(subf, module);
          subf = std::make_shared<SubFunction>(mode);
          insert_subop(subf, op);
        }
      }
    });
    if (subf != nullptr) {
      buildSubFunction(subf, module);
      subf = nullptr;
    }
    Module::removeUnusedOp(module);
    Module::setState(module, Module::State::TPU_DIVIDED);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createSubnetDividePass() {
  return std::make_unique<SubnetDividePass>();
}
} // namespace tpu
} // namespace tpu_mlir

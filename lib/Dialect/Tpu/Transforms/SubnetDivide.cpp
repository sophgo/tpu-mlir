//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements the TPU dialect OP Stats pass.
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Tpu/Transforms/Passes.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Backend/BM1684.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <sstream>
#include <fstream>
#include <set>

using namespace llvm;
using namespace mlir;
using namespace sophgo::helper;
using namespace sophgo::backend;
namespace sophgo {
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
  int64_t id = SubFunction::count;
  std::string func_name = "subfunc_" + std::to_string(id);
  OpBuilder builder(module.getContext());
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("id", builder.getI64IntegerAttr(id)));
  attrs.push_back(builder.getNamedAttr("mode", builder.getStringAttr(sf->mode)));
  auto fnType = builder.getFunctionType(llvm::ArrayRef<Type>{argType},
                                        llvm::ArrayRef<Type>{resType});
  auto fnOp = FuncOp::create(module.getLoc(), func_name, fnType,ArrayRef<NamedAttribute>(attrs));
  auto block = fnOp.addEntryBlock();
  builder.setInsertionPointAfterValue(fnOutputs.back());
  func::CallOp callOp = builder.create<func::CallOp>(module.getLoc(), func_name,
                                                     resType, fnInputs);
  for (auto it : llvm::enumerate(callOp.getResults())) {
    fnOutputs[it.index()].replaceUsesWithIf(it.value(), [&](OpOperand &operand) {
      Operation *user = operand.getOwner();
      return find(sf->ops.begin(), sf->ops.end(), user) == sf->ops.end();
    });
  }
  builder.setInsertionPointToStart(block);
  auto retOp = builder.create<func::ReturnOp>(module.getLoc(), fnOutputs);
  for (auto op : sf->ops) {
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
  if (isa<tpu::MaxPoolOp>(op)) {
    return FUNC_CPU; // here just simulate
  } else {
    return FUNC_TPU;
  }
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
      if (isa<top::InputOp>(op) || isa<top::WeightOp>(op) || isa<FuncOp>(op) ||
          isa<top::NoneOp>(op) || isa<func::ReturnOp>(op) ||
          isa<func::CallOp>(op)) {
        // do nothing
      } else {
        auto mode = getOpMode(op);
        if (subf == nullptr) {
          subf = std::make_shared<SubFunction>(mode);
          for (auto opd : op->getOperands()) {
            auto op_ = opd.getDefiningOp();
            if (isa<top::WeightOp>(op_)) {
              subf->ops.push_back(op_);
            } else if (isa<top::NoneOp>(op_) && subf->have_none == false) {
              subf->have_none = true;
              subf->ops.push_back(op_);
            }
          }
          subf->ops.push_back(op);
        } else if (subf->mode == mode) {
          for (auto opd : op->getOperands()) {
            auto op_ = opd.getDefiningOp();
            if (isa<top::WeightOp>(op_)) {
              subf->ops.push_back(op_);
            } else if (isa<top::NoneOp>(op_) && subf->have_none == false) {
              subf->have_none = true;
              subf->ops.push_back(op_);
            }
          }
          subf->ops.push_back(op);
        } else {
          buildSubFunction(subf, module);
          subf = std::make_shared<SubFunction>(mode);
          for (auto opd : op->getOperands()) {
            auto op_ = opd.getDefiningOp();
            if (isa<top::WeightOp>(op_)) {
              subf->ops.push_back(op_);
            } else if (isa<top::NoneOp>(op_) && subf->have_none == false) {
              subf->have_none = true;
              subf->ops.push_back(op_);
            }
          }
          subf->ops.push_back(op);
        }
      }
    });
    if (subf != nullptr) {
      buildSubFunction(subf, module);
      subf = nullptr;
    }
    Module::setState(module, Module::State::TPU_DIVIDED);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createSubnetDividePass() {
  return std::make_unique<SubnetDividePass>();
}
} // namespace tpu
} // namespace sophgo

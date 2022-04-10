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
#include "sophgo/Backend/BM168x/BM1684.h"
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

class AddressAsignPass : public AddressAsignBase<AddressAsignPass> {
public:
  AddressAsignPass() {}
  void runOnOperation() override {
    auto module = getOperation();
    auto state = Module::getState(module);
    if (state != Module::State::TPU_DIVIDED) {
      llvm_unreachable("module should be reordered");
    }
    int64_t start_addr = 0;
    int64_t alignment = 0;
    if (Module::getChip(module) == Module::Chip::BM1684) {
      start_addr = BM1684::instance().get_ctx_start_addr();
      alignment = BM1684::ALIGNMENT;
    } else {
      llvm_unreachable("chip not support now");
    }
    Builder builder(module.getContext());
    // asign weight first
    auto addr = start_addr;
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](top::WeightOp op) {
        Module::setAddress(op.output(), addr);
        int64_t bytes = Module::getBytes(op.output());
        addr += ALIGN(bytes, alignment);
      });
    }
    Module::setCoeffAddr(module, start_addr);
    Module::setCoeffSize(module, addr - start_addr);
    // asign activation
    start_addr = addr;
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        if (isa<FuncOp, top::NoneOp, func::ReturnOp, top::WeightOp,
                func::CallOp>(op)) {
        } else {
          auto output = op->getResult(0);
          Module::setAddress(output, addr);
          int64_t bytes = Module::getBytes(output);
          addr += ALIGN(bytes, alignment);
        }
      });
    }
    Module::setNeuronAddr(module, start_addr);
    Module::setNeuronSize(module, addr - start_addr);
    Module::updateModuleTypes(module);
    Module::setState(module, Module::State::TPU_ADDRESSED);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAddressAsignPass() {
  return std::make_unique<AddressAsignPass>();
}
} // namespace tpu
} // namespace sophgo

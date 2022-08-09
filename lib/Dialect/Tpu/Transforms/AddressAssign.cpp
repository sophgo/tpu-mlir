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
#include "tpu_mlir/Backend/BM168x/BM1684x.h"

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

class AddressAssignPass : public AddressAssignBase<AddressAssignPass> {
public:
  AddressAssignPass() {}
  void runOnOperation() override {
    auto module = getOperation();
    auto state = Module::getState(module);
    if (state != Module::State::TPU_DIVIDED) {
      llvm_unreachable("module should be divided");
    }
    Module::removeUnusedOp(module);
    int64_t start_addr = 0;
    int64_t alignment = BM168x::ALIGNMENT;
    chip = Module::getChip(module);
    if (chip == Module::Chip::BM1684) {
      start_addr = BM1684::instance().get_ctx_start_addr();
    } else if (chip == Module::Chip::BM1684x) {
      start_addr = BM1684x::instance().get_ctx_start_addr();
    } else {
      llvm_unreachable("chip not support now");
    }
    Builder builder(module.getContext());
    // assign weight first
    auto addr = start_addr;
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](top::WeightOp op) {
        Module::setAddress(op.output(), addr);
        int64_t bytes = Module::getBytes(op.output());
        addr = align_up(addr + bytes, alignment);
      });
    }
    Module::setCoeffAddr(module, start_addr);
    Module::setCoeffSize(module, addr - start_addr);
    // assign activation
    start_addr = addr;
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        if (isa<FuncOp, top::NoneOp, func::ReturnOp, top::WeightOp,
                func::CallOp, tpu::YieldOp>(op)) {
        } else if (fuse_address(op)) {
          // do nothing
        } else {
          for (auto out : op->getResults()) {
            Module::setAddress(out, addr);
            int64_t bytes = Module::getBytes(out);
            addr = align_up(addr + bytes, alignment);
          }
        }
      });
      // sync StoreOp addr
      func.walk([&](tpu::GroupOp gOp) {
        int idx = 0;
        gOp.body().walk([&](tpu::StoreOp sOp) {
          auto addr = Module::getAddress(gOp.getResult(idx));
          Module::setAddress(sOp.output(), addr);
          idx++;
        });
      });
    }
    Module::setNeuronAddr(module, start_addr);
    Module::setNeuronSize(module, addr - start_addr);
    Module::updateModuleTypes(module);
    Module::setState(module, Module::State::TPU_ADDRESSED);
  }

protected:
  bool fuse_address(Operation *op) {
    if (Module::isOpInGroup(op)) {
      return true;
    }
    if (auto reshapeOp = dyn_cast<tpu::ReshapeOp>(op)) {
      if (chip == Module::Chip::BM1684x) {
        auto addr = Module::getAddress(reshapeOp.input());
        Module::setAddress(reshapeOp.output(), addr);
        return true;
      }
    }
    return false;
  }
  StringRef chip;
};

std::unique_ptr<OperationPass<ModuleOp>> createAddressAssignPass() {
  return std::make_unique<AddressAssignPass>();
}
} // namespace tpu
} // namespace tpu_mlir

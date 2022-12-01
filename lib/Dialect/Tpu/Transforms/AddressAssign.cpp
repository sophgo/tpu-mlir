//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/BMAddressAssign.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/CVAddressAssign.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

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
    auto chip = Module::getChip(module);
    Arch::init(chip);

    if (Module::isCV18xx(chip)) {
      CVAddressAssign addr_assign;
      addr_assign.assign(module);
    } else {
      RewritePatternSet patterns(module.getContext());
      bm168x::populateGlobalBufferPatterns(&patterns);
      applyPatternsAndFoldGreedily(module, std::move(patterns));
      BMAddressAssign addr_assign;
      addr_assign.assign(module);
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAddressAssignPass() {
  return std::make_unique<AddressAssignPass>();
}
} // namespace tpu
} // namespace tpu_mlir

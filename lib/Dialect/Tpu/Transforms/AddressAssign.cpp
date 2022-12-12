//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/BMAddressAssign.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/CVAddressAssign.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {
class AddressAssignPass : public AddressAssignBase<AddressAssignPass> {
public:
  AddressAssignPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    if (!module::isState(module::State::TPU_DIVIDED)) {
      llvm_unreachable("module should be divided");
    }
    module::removeUnusedOp();
    Arch::init();

    if (module::isCV18xx()) {
      CVAddressAssign addr_assign;
      addr_assign.assign(mOp, reuse_addr);
    } else {
      auto bm168x = backend::BM168x::instance();
      bm168x->start_env();
      RewritePatternSet patterns(mOp.getContext());
      bm168x::populateGlobalBufferPatterns(&patterns);
      applyPatternsAndFoldGreedily(mOp, std::move(patterns));
      BMAddressAssign addr_assign;
      addr_assign.assign(mOp, reuse_addr);
      bm168x->end_env();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAddressAssignPass() {
  return std::make_unique<AddressAssignPass>();
}
} // namespace tpu
} // namespace tpu_mlir

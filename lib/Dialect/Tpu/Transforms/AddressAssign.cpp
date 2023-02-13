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

class ConcatFusePattern : public OpRewritePattern<tpu::ConcatOp> {
public:
  using OpRewritePattern<tpu::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::ConcatOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getOnlyMerge()) {
      return failure();
    }
    if (op.getDoRelu()) {
      return failure();
    }
    auto shape = module::getShape(op.getOutput());
    int outer_dim = std::accumulate(shape.begin(), shape.begin() + op.getAxis(),
                                    1, std::multiplies<int64_t>());
    if (outer_dim != 1) {
      return failure();
    }
    for (auto in : op.getInputs()) {
      if (module::isWeight(in)) {
        return failure();
      }
      if (in.hasOneUse() == false) {
        return failure();
      }
      auto in_op = in.getDefiningOp();
      if (in_op == nullptr) {
        return failure();
      } else if (isa<tpu::ConcatOp>(in_op)) {
        return failure();
      } else if (auto rshape = dyn_cast<tpu::ReshapeOp>(in_op)) {
        auto in2 = rshape.getInput();
        if (in2.getDefiningOp() == nullptr || in2.hasOneUse() == false) {
          return failure();
        }
      } else if (auto sliceOp = dyn_cast<tpu::SliceOp>(in_op)) {
        auto p = sliceOp.parseParam();
        if (p.fusible) {
          return failure();
        }
      }
    }
    op.setOnlyMerge(true);
    return success();
  }
};

class AddressAssignPass : public AddressAssignBase<AddressAssignPass> {
public:
  AddressAssignPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    if (!module::isState(module::State::TPU_DIVIDED)) {
      llvm_unreachable("module should be divided");
    }
    module::removeUnusedOp();
    if (module::isCV18xx()) {
      CVAddressAssign addr_assign;
      addr_assign.assign(mOp, reuse_addr);
    } else {
      RewritePatternSet patterns(mOp.getContext());
      bm168x::populateGlobalBufferPatterns(&patterns);
      patterns.add<ConcatFusePattern>(patterns.getContext());
      applyPatternsAndFoldGreedily(mOp, std::move(patterns));
      BMAddressAssign addr_assign;
      addr_assign.assign(mOp, reuse_addr);
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAddressAssignPass() {
  return std::make_unique<AddressAssignPass>();
}
} // namespace tpu
} // namespace tpu_mlir

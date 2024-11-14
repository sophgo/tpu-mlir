//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;

// for bert model, concat matmuls to batch matmul
struct PackMatmulPattern : public OpRewriterPatternEx<PackOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  PackMatmulPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<PackOp>(context, "OpRewriterPatternEx") {}

  LogicalResult matchAndRewriteImpl(PackOp op,
                                    PatternRewriter &rewriter) const override {
    if (!op->hasOneUse()) {
      return failure();
    }
    Value l_in = nullptr;
    Value r_in = nullptr;
    for (auto in : op.getInputs()) {
      auto matmul = dyn_cast<MatMulOp>(in.getDefiningOp());
      if (!matmul || !matmul->hasOneUse()) {
        return failure();
      }
      auto the_slice_op = matmul.getInput().getDefiningOp();
      auto slice = dyn_cast<SliceOp>(the_slice_op);
      if (!slice || !slice->hasOneUse()) {
        return failure();
      } else {
        auto steps = module::getI64Array(slice.getSteps());
        auto step_ = std::accumulate(steps->begin(), steps->end(), 1,
                                     std::multiplies<int64_t>());
        if (step_ != 1) {
          return failure();
        }
        auto shape = module::getShape(slice.getOutput());
        if (shape[0] != 1) {
          return failure();
        }
      }
      if (l_in == nullptr) {
        l_in = slice.getInput();
      } else if (l_in != slice.getInput()) {
        return failure();
      }
      auto permute = dyn_cast<PermuteOp>(matmul.getRight().getDefiningOp());
      if (!permute || !permute->hasOneUse()) {
        return failure();
      }
      auto reshape = dyn_cast<ReshapeOp>(permute.getInput().getDefiningOp());
      if (!reshape || !reshape->hasOneUse()) {
        return failure();
      }
      the_slice_op = reshape.getInput().getDefiningOp();
      slice = dyn_cast<SliceOp>(the_slice_op);
      if (!slice || !slice->hasOneUse()) {
        return failure();
      } else {
        auto steps = module::getI64Array(slice.getSteps());
        auto step_ = std::accumulate(steps->begin(), steps->end(), 1,
                                     std::multiplies<int64_t>());
        if (step_ != 1) {
          return failure();
        }
        auto shape = module::getShape(slice.getOutput());
        if (shape[0] != 1) {
          return failure();
        }
      }
      if (r_in == nullptr) {
        r_in = slice.getInput();
      } else if (r_in != slice.getInput()) {
        return failure();
      }
    }
    if (r_in == nullptr || l_in == nullptr) {
      return failure();
    }
    rewriter.setInsertionPointAfter(op);
    auto none = module::getNoneOp(op);
    std::vector<NamedAttribute> attrs;
    auto batchMatMul =
        rewriter.create<MatMulOp>(op.getLoc(), op.getOutput().getType(),
                                  ValueRange{l_in, r_in, none}, attrs);
    op.replaceAllUsesWith(batchMatMul.getOperation());
    rewriter.eraseOp(op);
    return success();
  }
};

void PackOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<PackMatmulPattern>(context);
}

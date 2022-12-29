//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"

#include "tpu_mlir/Support/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace tpu_mlir::top;


struct TopPermuteToPixelShuffle : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto input_shape = module::getShape(op.input());
    if (input_shape.size() != 6) {
      return failure();
    }

    std::vector<int64_t> ps = {0, 1, 4, 2, 5, 3};
    auto order = module::getI64Array(op.order());
    if (*order != ps) {
      return failure();
    }
    auto reshape_before = dyn_cast_or_null<ReshapeOp>(op.input().getDefiningOp());
    if (!reshape_before) {
      return failure();
    }
    auto nextOp = *op.output().getUsers().begin();
    auto reshape_after = dyn_cast_or_null<ReshapeOp>(nextOp);
    if (!reshape_after) {
      return failure();
    }
    auto output_shape = module::getShape(reshape_after.output());
    int64_t upscale_factor = input_shape[2];
    int64_t on = input_shape[0];
    int64_t oc = input_shape[1];
    int64_t oh = upscale_factor * input_shape[4];
    int64_t ow = upscale_factor * input_shape[5];
    std::vector<int64_t> o_s = {on, oc, oh, ow};
    if (output_shape.vec() != o_s) {
      return failure();
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("is_CRD", rewriter.getBoolAttr(true)));
    attrs.push_back(
        rewriter.getNamedAttr("is_inversed", rewriter.getBoolAttr(false)));
    attrs.push_back(rewriter.getNamedAttr(
        "block_h", rewriter.getI64IntegerAttr(upscale_factor)));
    attrs.push_back(rewriter.getNamedAttr(
        "block_w", rewriter.getI64IntegerAttr(upscale_factor)));
    rewriter.replaceOpWithNewOp<Depth2SpaceOp>(reshape_after, reshape_after.getResult().getType(), ValueRange{reshape_before.input()}, attrs);
    rewriter.eraseOp(op);
    rewriter.eraseOp(reshape_before);
    return success();
  }
};

void PermuteOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<TopPermuteToPixelShuffle>(context);
}

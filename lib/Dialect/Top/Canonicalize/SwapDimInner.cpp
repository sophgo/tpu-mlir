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

struct TopMultiSwapDimMergeToOne : public OpRewriterPatternEx<SwapDimInnerOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopMultiSwapDimMergeToOne(mlir::MLIRContext *context)
      : OpRewriterPatternEx<SwapDimInnerOp>(context,
                                            "TopMultiSwapDimMergeToOne") {}

  LogicalResult matchAndRewriteImpl(SwapDimInnerOp op,
                                    PatternRewriter &rewriter) const override {
    auto nextOp = *op->user_begin();
    if (!isa<SwapDimInnerOp>(nextOp)) {
      return failure();
    }

    auto next_swap_op = dyn_cast<SwapDimInnerOp>(nextOp);
    auto cur_offset = module::getI64Array(op.getOffset());
    auto next_offset = module::getI64Array(next_swap_op.getOffset());
    assert(cur_offset->size() == next_offset->size());

    std::vector<int64_t> offset(cur_offset->size(), 0);
    int32_t axis_num = 0;
    for (size_t i = 0; i < cur_offset->size(); ++i) {
      if (cur_offset->at(i) != 0) {
        axis_num++;
        offset[i] = cur_offset->at(i);
      } else if (next_offset->at(i) != 0) {
        axis_num++;
        offset[i] = next_offset->at(i);
      }
    }
    // backend doesn't support axis_num > 2 yet
    if (axis_num > 2) {
      return failure();
    }

    nextOp->setAttr("offset", rewriter.getI64ArrayAttr(offset));
    rewriter.replaceOp(op, {op.getInput()});
    return success();
  }
};

void SwapDimInnerOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<TopMultiSwapDimMergeToOne>(context);
}

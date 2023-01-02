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

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct MergeSliceOp : public OpRewritePattern<SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp op,
                                PatternRewriter &rewriter) const override {

    auto in_op = op.getInput().getDefiningOp();
    if (!isa<SliceOp>(in_op) || in_op->hasOneUse() == false) {
      return failure();
    }
    auto output_shape = module::getShape(op.getOutput());
    auto num_dims = output_shape.size();
    auto in_slice = cast<SliceOp>(in_op);
    auto cur_offset = module::getI64Array(op.getOffset());
    auto cur_steps = module::getI64Array(op.getSteps());
    auto in_offset = module::getI64Array(in_slice.getOffset());
    auto in_steps = module::getI64Array(in_slice.getSteps());

    std::vector<int64_t> new_offset(num_dims, 0);
    std::vector<int64_t> new_steps(num_dims, 1);
    for (int i =0; i < num_dims; i++) {
        auto cur_off = cur_offset->at(i);
        auto cur_s = cur_steps->at(i);
        assert(cur_s > 0);
        auto in_off = in_offset->at(i);
        auto in_s = in_steps->at(i);
        assert(in_s > 0);
        new_offset[i] = in_off + cur_off * in_s;
        new_steps[i] = in_s * cur_s;
    }
    op->setAttr("offset", rewriter.getI64ArrayAttr(new_offset));
    op->setAttr("steps", rewriter.getI64ArrayAttr(new_steps));
    op->setOperand(0, in_slice.getInput());
    in_op->erase();
    return success();
  }
};

void SliceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<MergeSliceOp>(context);
}

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;

// squeeze + unsqueeze && in == out
struct TopFuseUnsqueeze : public OpRewritePattern<UnsqueezeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnsqueezeOp op,
                                PatternRewriter &rewriter) const override {
    auto in_op = op.getInput().getDefiningOp();

    if (in_op->hasOneUse() && isa<SqueezeOp>(in_op)) {
      auto former_op = dyn_cast<SqueezeOp>(in_op);
      auto shape0 = module::getShape(op.getOutput());
      auto shape1 = module::getShape(former_op.getInput());
      if (shape0 != shape1) {
        return failure();
      }
      op.getOutput().replaceAllUsesWith(former_op.getInput());
      rewriter.eraseOp(op);
      rewriter.eraseOp(former_op);
      return success();
    }
    return failure();
  }
};

struct TopGatherToSliceByUnsqueeze : public OpRewritePattern<GatherOp> {
  using OpRewritePattern::OpRewritePattern;
  TopGatherToSliceByUnsqueeze(MLIRContext *context)
      : OpRewritePattern<GatherOp>(context) {}

  LogicalResult matchAndRewrite(GatherOp op,
                                PatternRewriter &rewriter) const override {
    std::shared_ptr<std::vector<float>> inds_f32;

    if (auto inds = dyn_cast<WeightOp>(op.getIndices().getDefiningOp()))
      inds_f32 = inds.read<float>();
    else
      return failure();

    auto inds_elems = module::getNumElements(op.getIndices());
    auto ax = op.getAxis();
    // if indices are regular, try to convert to SliceOp
    if (inds_elems == 1) {
      // e.g. Gather(indices=[1],axis=ax) + Unsqueeze(axis=ax)
      //            -> Slice(start=1, end=2, step=1, axes=ax)
      auto nextOp = op->user_begin();
      if (!op->hasOneUse() || !isa<UnsqueezeOp>(*nextOp)) {
        // tmp code convert scalar to tensor for rangeOp (wait for shute's
        // commit)
        if (!isa<UnrankedTensorType>(op.getType()) &&
            module::getShape(op.getOutput()).size() == 0) {
          module::setShape(op.getResult(), {1});
        }
        for (auto user: op->getUsers()) {
          for (auto res: user->getResults()) {
            if (!isa<UnrankedTensorType>(res.getType()) && module::getShape(res).size() == 0)
            module::setShape(res, {1});
          }
        }
        return failure();
      }

      auto reshape_op = cast<UnsqueezeOp>(*nextOp);
      auto out_shape = module::getShape(op.getOutput());
      auto reshape_out_shape = module::getShape(reshape_op.getOutput());
      std::vector<int64_t> unsqueeze_out_shape{};
      for (int64_t i = 0; i < out_shape.size(); ++i) {
        if (i == ax && !op.getKeepdims()) {
          unsqueeze_out_shape.push_back(1);
        }
        unsqueeze_out_shape.push_back(out_shape[i]);
      }
      if (out_shape.size() == 0 && !op.getKeepdims()) {
        unsqueeze_out_shape.push_back(1);
      }
      if (unsqueeze_out_shape.size() != reshape_out_shape.size()) {
        return failure();
      }
      for (int64_t i = 0; i < unsqueeze_out_shape.size(); ++i) {
        if (unsqueeze_out_shape[i] != reshape_out_shape[i]) {
          return failure();
        }
      }

      NamedAttrList attrs;
      auto none = module::getNoneOp(op);
      auto input_shape = module::getShape(op.getInput());
      std::vector<int64_t> offsets(input_shape.size(), 0);
      std::vector<int64_t> steps(input_shape.size(), 1);
      std::vector<int64_t> ends(input_shape.size(), 1);
      offsets[ax] = (int64_t)inds_f32->at(0);
      int64_t step = offsets[ax] - steps[ax];
      if (step <= 0)
        return failure();
      ends[ax] = input_shape[ax];
      attrs.set("offset", rewriter.getI64ArrayAttr(offsets));
      attrs.set("steps", rewriter.getI64ArrayAttr(steps));
      attrs.set("ends", rewriter.getI64ArrayAttr(ends));
      op.getOperation()->setLoc(reshape_op.getLoc());
      rewriter.replaceOpWithNewOp<SliceOp>(
          op, reshape_op.getOutput().getType(),
          ValueRange{op.getInput(), none, none, none}, attrs);
      rewriter.replaceOp(reshape_op, {reshape_op.getInput()});
      return success();
    }
    // replace the Gather Op and remove the next UnsqueezeOp
    return failure();
  }
};
void UnsqueezeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<TopFuseUnsqueeze, TopGatherToSliceByUnsqueeze>(context);
}

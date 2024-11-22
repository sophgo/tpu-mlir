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

// remove: reshape + unsqueeze && in == out
struct TopRemoveReshapeAndUnsqueezeWhenScalar
    : public OpRewriterPatternEx<UnsqueezeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  TopRemoveReshapeAndUnsqueezeWhenScalar(mlir::MLIRContext *context)
      : OpRewriterPatternEx<UnsqueezeOp>(
            context, "TopRemoveReshapeAndUnsqueezeWhenScalar") {}

  LogicalResult matchAndRewriteImpl(UnsqueezeOp op,
                                    PatternRewriter &rewriter) const override {
    auto in_op = op.getInput().getDefiningOp();

    if (in_op->hasOneUse() && isa<ReshapeOp>(in_op)) {
      auto former_op = dyn_cast<ReshapeOp>(in_op);
      auto out_shape = module::getShape(op.getOutput());
      auto in_shape = module::getShape(former_op.getInput());
      if (in_shape != out_shape) {
        return failure();
      }
      op.getOutput().replaceAllUsesWith(former_op.getInput());
      auto former_former_op = former_op.getInput().getDefiningOp();
      if (!isa<top::InputOp>(former_former_op)) {
        former_former_op->setLoc(op.getLoc());
      }
      rewriter.eraseOp(op);
      rewriter.eraseOp(former_op);
      return success();
    }
    return failure();
  }
};

// squeeze + unsqueeze && in == out
struct TopFuseUnsqueeze : public OpRewriterPatternEx<UnsqueezeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopFuseUnsqueeze(mlir::MLIRContext *context)
      : OpRewriterPatternEx<UnsqueezeOp>(context, "TopFuseUnsqueeze") {}

  LogicalResult matchAndRewriteImpl(UnsqueezeOp op,
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
      auto former_former_op = former_op.getInput().getDefiningOp();
      if (!isa<top::InputOp>(former_former_op)) {
        former_former_op->setLoc(op.getLoc());
      }
      rewriter.eraseOp(op);
      rewriter.eraseOp(former_op);
      return success();
    }
    return failure();
  }
};

// void recursivelyModifyShapes(Operation *op, PatternRewriter &rewriter) {
//   if (!isa<UnrankedTensorType>(op->getResult(0).getType()) &&
//       module::getShape(op->getResult(0)).size() == 0) {
//     module::setShape(op->getResult(0), {1});
//     if (op->hasTrait<trait::ScalarProducer>()) {
//       op->setAttr("is_scalar", rewriter.getBoolAttr(true));
//     }
//     if (op->getResult(0).getUsers().empty()) {
//       return;
//     }
//     for (auto user : op->getResult(0).getUsers()) {
//       recursivelyModifyShapes(user, rewriter);
//     }
//   }
//   return;
// }

struct TopGatherToSliceByUnsqueeze : public OpRewriterPatternEx<GatherOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  TopGatherToSliceByUnsqueeze(MLIRContext *context)
      : OpRewriterPatternEx<GatherOp>(context, "TopGatherToSliceByUnsqueeze") {}

  LogicalResult matchAndRewriteImpl(GatherOp op,
                                    PatternRewriter &rewriter) const override {
    std::shared_ptr<std::vector<float>> inds_f32;

    if (auto inds = dyn_cast<WeightOp>(op.getIndices().getDefiningOp()))
      inds_f32 = inds.read_as_float();
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
        // if (!isa<UnrankedTensorType>(op.getType()) &&
        //     module::getShape(op.getOutput()).size() == 0) {
        //   module::setShape(op.getResult(), {1});
        // }
        // for (auto user: op->getUsers()) {
        //   for (auto res: user->getResults()) {
        //     if (!isa<UnrankedTensorType>(res.getType()) &&
        //     module::getShape(res).size() == 0) module::setShape(res, {1});
        //   }
        // }
        // recursivelyModifyShapes(op.getOperation(), rewriter);
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
      std::vector<int64_t> ends(input_shape.size(),
                                std::numeric_limits<int64_t>::max());
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

// unsqueeze scalar [1] -> [1]
struct TopUnsqueezeErase : public OpRewriterPatternEx<UnsqueezeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopUnsqueezeErase(MLIRContext *context)
      : OpRewriterPatternEx<UnsqueezeOp>(context, "TopUnsqueezeErase") {}

  LogicalResult matchAndRewriteImpl(UnsqueezeOp op,
                                    PatternRewriter &rewriter) const override {
    auto shape0 = module::getShape(op.getOutput());
    auto shape1 = module::getShape(op.getInput());
    if (shape0 != shape1) {
      return failure();
    }
    op.getOutput().replaceAllUsesWith(op.getInput());
    auto former_op = op.getInput().getDefiningOp();
    if (!isa<top::InputOp>(former_op)) {
      former_op->setLoc(op.getLoc());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

void UnsqueezeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<TopRemoveReshapeAndUnsqueezeWhenScalar, TopFuseUnsqueeze,
                 TopGatherToSliceByUnsqueeze, TopUnsqueezeErase>(context);
}

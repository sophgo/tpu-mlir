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
using namespace tpu_mlir::trait;

struct TopGatherToSlice : public OpRewriterPatternEx<GatherOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopGatherToSlice(MLIRContext *context)
      : OpRewriterPatternEx<GatherOp>(context,"TopGatherToSlice") {}

  LogicalResult matchAndRewriteImpl(GatherOp op,
                                PatternRewriter &rewriter) const override {
    std::shared_ptr<std::vector<float>> inds_f32;

    if (auto inds = dyn_cast<WeightOp>(op.getIndices().getDefiningOp()))
      inds_f32 = inds.read_as_float();
    else
      return failure();

    auto inds_shape = module::getShape(op.getIndices());
    auto inds_elems = module::getNumElements(op.getIndices());
    auto ax = op.getAxis();
    auto input_shape = module::getShape(op.getInput());
    // if indices are regular, try to convert to SliceOp
    if (inds_elems == 1) {
      // e.g. Gather(indices=[1],axis=ax) + Unsqueeze(axis=ax)
      //            -> Slice(start=1, end=2, step=1, axes=ax)
      int64_t index = (int64_t)inds_f32->at(0);
      if (index < 0) {
        if (index != -1) {
          index = input_shape[ax] + index;
        }
      }
      std::vector<int64_t> offsets(input_shape.size(), 0);
      std::vector<int64_t> ends(input_shape.size(), std::numeric_limits<int64_t>::max());
      std::vector<int64_t> steps(input_shape.size(), 1);
      offsets[ax] = index;
      ends[ax] = offsets[ax] + 1;

      NamedAttrList attrs;
      attrs.set("offset", rewriter.getI64ArrayAttr(offsets));
      attrs.set("steps", rewriter.getI64ArrayAttr(steps));
      attrs.set("ends", rewriter.getI64ArrayAttr(ends));
      auto none = module::getNoneOp(op);
      std::vector<Value> operands;
      operands.push_back(op.getInput());
      operands.push_back(none);
      operands.push_back(none);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      std::vector<int64_t> slice_shape = input_shape;
      slice_shape[ax] = 1;
      auto new_type = module::getTypeLike(op.getOutput(), slice_shape);
      auto new_loc = module::getLocLike(op.getOutput(), "slice");
      auto slice_op =
          rewriter.create<SliceOp>(new_loc, new_type, operands, attrs);
      if (slice_shape.size() != 1) {
        std::vector<int64_t> axes(1, ax);
        std::vector<NamedAttribute> squeeze_attrs;
        squeeze_attrs.push_back(rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr(axes)));
        auto squeeze_op = rewriter.create<SqueezeOp>(
            op.getLoc(), op.getType(), ValueRange{slice_op.getOutput()}, squeeze_attrs);
        rewriter.replaceOp(op, {squeeze_op.getOutput()});
      } else {
        rewriter.replaceOp(op, {slice_op.getOutput()});
      }
      return success();
    } else if (inds_shape.size() == 1) {
      // e.g. Gather(indices=[1,3,5,7],axis=ax)
      //            -> Slices(start=1, end=8, step=2, axes=ax)
      int64_t offset = (int64_t)inds_f32->at(0);
      int64_t step = (int64_t)inds_f32->at(1) - offset;
      for (int i = 2; i < inds_shape[0]; ++i) {
        if (offset + i * step != inds_f32->at(i)) {
          return failure();
        }
      }
      if (step <= 0)
        return failure();

      NamedAttrList attrs;
      auto input_shape = module::getShape(op.getInput());
      std::vector<int64_t> offsets(input_shape.size(), 0);
      std::vector<int64_t> ends(input_shape.size(), -1);
      std::vector<int64_t> steps(input_shape.size(), 1);
      offsets[ax] = (int64_t)inds_f32->at(0);
      steps[ax] = step;
      ends[ax] = offsets[ax] + steps[ax] * (inds_shape[0] - 1);
      attrs.set("offset", rewriter.getI64ArrayAttr(offsets));
      attrs.set("steps", rewriter.getI64ArrayAttr(steps));
      attrs.set("ends", rewriter.getI64ArrayAttr(ends));
      auto none = module::getNoneOp(op);
      std::vector<Value> operands;
      operands.push_back(op.getInput());
      operands.push_back(none);
      operands.push_back(none);
      operands.push_back(none);
      rewriter.replaceOpWithNewOp<SliceOp>(op, op.getOutput().getType(),
                                           operands, attrs);
      return success();
    }

    // replace the Gather Op and remove the next reshapeOp
    return failure();
  }
};

struct TopGatherMulConst : public OpRewriterPatternEx<GatherOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  TopGatherMulConst(MLIRContext *context)
      : OpRewriterPatternEx<GatherOp>(context,"TopGatherMulConst") {}

  LogicalResult matchAndRewriteImpl(GatherOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasOneUse() || !module::isWeight(op.getInput())) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
      return failure();
    }
    auto next_op = *op->user_begin();
    auto mulconst_op = dyn_cast<top::MulConstOp>(next_op);
    if (!mulconst_op) {
      return failure();
    }
    auto weight_op = cast<top::WeightOp>(op.getInput().getDefiningOp());
    auto data = weight_op.read_as_float();
    auto data_new = std::make_shared<std::vector<float>>(data->size());
    std::transform(data->begin(), data->end(), data_new->begin(),
                   [&](const float d) {
                     return d * mulconst_op.getConstVal().convertToDouble();
                   });
    auto w_shape = module::getShape(op.getInput());
    auto new_weight =
        WeightOp::create_float(op, "merge_mulconst", *data_new, w_shape, storage_type);
    op.setOperand(0, new_weight);
    op->setLoc(mulconst_op->getLoc());
    rewriter.replaceOp(mulconst_op, op);
    return success();
  }
};

void GatherOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<TopGatherToSlice, TopGatherMulConst>(context);
}

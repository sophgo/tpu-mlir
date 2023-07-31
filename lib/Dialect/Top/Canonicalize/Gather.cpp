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
using namespace tpu_mlir::trait;

struct TopGatherToSlice : public OpRewritePattern<GatherOp> {
  using OpRewritePattern::OpRewritePattern;
  TopGatherToSlice(MLIRContext *context)
      : OpRewritePattern<GatherOp>(context) {}

  LogicalResult matchAndRewrite(GatherOp op,
                                PatternRewriter &rewriter) const override {
    std::shared_ptr<std::vector<float>> inds_f32;

    if (auto inds = dyn_cast<WeightOp>(op.getIndices().getDefiningOp()))
      inds_f32 = inds.read<float>();
    else
      return failure();

    auto inds_shape = module::getShape(op.getIndices());
    auto inds_elems = module::getNumElements(op.getIndices());
    auto ax = op.getAxis();
    // if indices are regular, try to convert to SliceOp
    if (inds_elems == 1) {
      // e.g. Gather(indices=[1],axis=ax) + Unsqueeze(axis=ax)
      //            -> Slice(start=1, end=2, step=1, axes=ax)
      auto nextOp = op->user_begin();
      if (!op->hasOneUse()) {
        return failure();
      }
      bool next_is_reshape = false;
      if (!isa<ReshapeOp>(*nextOp)) {
        // e.g. Gather(indices=[1],axis=ax)
        //            -> Slice(start=1, end=2, step=1, axes=ax)
        next_is_reshape = true;
      }
      auto reshape_op = dyn_cast<ReshapeOp>(*nextOp);
      auto out_shape = module::getShape(op.getOutput());
      std::vector<int64_t> unsqueeze_out_shape{};
      for (int64_t i = 0; i < out_shape.size(); ++i) {
        if (i == ax && !op.getKeepdims()) {
          unsqueeze_out_shape.push_back(1);
        }
        unsqueeze_out_shape.push_back(out_shape[i]);
      }
      if (!next_is_reshape) {
        auto reshape_out_shape = module::getShape(reshape_op.getOutput());
        if (unsqueeze_out_shape.size() != reshape_out_shape.size()) {
          return failure();
        }
        for (int64_t i = 0; i < unsqueeze_out_shape.size(); ++i) {
          if (unsqueeze_out_shape[i] != reshape_out_shape[i]) {
            return failure();
          }
        }
      }
      NamedAttrList attrs;
      auto input_shape = module::getShape(op.getInput());
      std::vector<int64_t> offsets(input_shape.size(), 0);
      std::vector<int64_t> ends(input_shape.size(), -1);
      std::vector<int64_t> steps(input_shape.size(), 1);
      offsets[ax] = (int64_t)inds_f32->at(0);
      ends[ax] = offsets[ax] + 1;
      attrs.set("offset", rewriter.getI64ArrayAttr(offsets));
      attrs.set("steps", rewriter.getI64ArrayAttr(steps));
      attrs.set("ends", rewriter.getI64ArrayAttr(ends));
      auto none = module::getNoneOp(op);
      std::vector<Value> operands;
      operands.push_back(op.getInput());
      operands.push_back(none);
      operands.push_back(none);
      operands.push_back(none);
      if (!next_is_reshape) {
        op.getOperation()->setLoc(reshape_op.getLoc());
        rewriter.replaceOpWithNewOp<SliceOp>(
            op, reshape_op.getOutput().getType(), operands, attrs);
        rewriter.replaceOp(reshape_op, {reshape_op.getInput()});
      } else {
        rewriter.setInsertionPoint(op);
        auto name = module::getName(op.getOutput()).str();
        auto loc = NameLoc::get(rewriter.getStringAttr(name + "_slice"));
        auto slice_shape = unsqueeze_out_shape;
        if (input_shape.size() == 1 && out_shape.size() == 1) {
          slice_shape = {1};
        }
        auto new_type = RankedTensorType::get(slice_shape,
                                              module::getElementType(op));
        auto slice_op =
            rewriter.create<SliceOp>(loc, new_type, operands, attrs);

        auto reshape_op = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(),
                                                     ValueRange{slice_op});
        op.replaceAllUsesWith(reshape_op.getOperation());
        rewriter.eraseOp(op);
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
      if (step == 0)
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

void GatherOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<TopGatherToSlice>(context);
}

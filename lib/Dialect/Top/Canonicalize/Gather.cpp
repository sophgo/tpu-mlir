//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;
using namespace tpu_mlir::helper;

struct TopGatherToSlice : public OpRewritePattern<GatherOp> {
  using OpRewritePattern::OpRewritePattern;
  TopGatherToSlice(MLIRContext *context)
      : OpRewritePattern<GatherOp>(context) {}

  LogicalResult matchAndRewrite(GatherOp op,
                                PatternRewriter &rewriter) const override {

    auto inds = cast<WeightOp>(op.indices().getDefiningOp());
    auto inds_f32 = inds.read<float>();
    auto inds_shape = Module::getShape(op.indices());
    auto inds_elems = Module::getNumElements(op.indices());
    auto ax = op.axis();
    // if indices are regular, try to convert to SliceOp
    if (inds_elems == 1) {
      // e.g. Gather(indices=[1],axis=ax) + Unsqueeze(axis=ax)
      //            -> Slice(start=1, end=2, step=1, axes=ax)
      auto nextOp = op->getUsers().begin();
      if (!op->hasOneUse() || !isa<ReshapeOp>(*nextOp)) {
        return failure();
      }

      auto reshape_op = cast<ReshapeOp>(*nextOp);
      auto out_shape = Module::getShape(op.output());
      auto reshape_out_shape = Module::getShape(reshape_op.output());
      std::vector<int64_t> unsqueeze_out_shape{};
      for (int64_t i = 0; i < out_shape.size(); ++i) {
        if (i == ax) {
          unsqueeze_out_shape.push_back(1);
        }
        unsqueeze_out_shape.push_back(out_shape[i]);
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
      auto input_shape = Module::getShape(op.input());
      std::vector<int64_t> offsets(input_shape.size(), 0);
      std::vector<int64_t> steps(input_shape.size(), 1);
      offsets[ax] = (int64_t)inds_f32->at(0);
      attrs.set("offset", rewriter.getI64ArrayAttr(offsets));
      attrs.set("steps", rewriter.getI64ArrayAttr(steps));
      op.getOperation()->setLoc(reshape_op.getLoc());
      rewriter.replaceOpWithNewOp<SliceOp>(op, reshape_op.output().getType(),
                                           ValueRange{op.input()}, attrs);
      rewriter.replaceOp(reshape_op, {reshape_op.input()});
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

      NamedAttrList attrs;
      auto input_shape = Module::getShape(op.input());
      std::vector<int64_t> offsets(input_shape.size(), 0);
      std::vector<int64_t> steps(input_shape.size(), 1);
      offsets[ax] = (int64_t)inds_f32->at(0);
      steps[ax] = step;
      attrs.set("offset", rewriter.getI64ArrayAttr(offsets));
      attrs.set("steps", rewriter.getI64ArrayAttr(steps));
      rewriter.replaceOpWithNewOp<SliceOp>(op, op.output().getType(),
                                           ValueRange{op.input()}, attrs);
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

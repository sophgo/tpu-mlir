//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir::top;

struct TopRepeatToTile : public OpRewritePattern<RepeatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RepeatOp op,
                                PatternRewriter &rewriter) const override {

    auto out_shape = module::getShape(op.getOutput());
    auto in_shape = module::getShape(op.getInput());
    auto in_shape_ = shape_expand_dim(in_shape, out_shape.size());
    auto last_shape = std::vector<int64_t>(in_shape_);
    auto stype = module::getStorageType(op.getInput());
    // auto repeats_ = module::getI64Array(getRepeatsAttr());
    int last_i = 0;
    auto last_op = op.getInput();
    if (in_shape.size() < out_shape.size()) {
      std::vector<NamedAttribute> attrs;
      auto newType = RankedTensorType::get(in_shape_, stype);
      auto new_name = module::getName(op.getOperation()).str() + "_reshape";
      auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
      last_op = rewriter.create<ReshapeOp>(
          name_loc, newType, ValueRange{last_op}, attrs);
    }
    for (int i = out_shape.size() - 1; i >= 0; --i) {
      last_i = i;
      if (in_shape_[last_i] != out_shape[last_i])
        break;
    }
    if (last_i == 0 && in_shape_[last_i] == out_shape[last_i]) {
      op.getOutput().replaceAllUsesWith(op.getInput());
      return success();
    }
    for (int i = 0; i <= last_i; ++i) {
      if (in_shape_[i] == out_shape[i])
        continue;
      int64_t tile = out_shape[i] / in_shape_[i];
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(i)));
      attrs.push_back(
          rewriter.getNamedAttr("tile", rewriter.getI64IntegerAttr(tile)));

      if (i == last_i) {
        rewriter.replaceOpWithNewOp<TileOp>(
            op, op.getResult().getType(), ValueRange{last_op}, attrs);
      } else {
        last_shape[i] = out_shape[i];
        auto newType = RankedTensorType::get(last_shape, stype);
        auto new_name = module::getName(op.getOperation()).str() + "___" +
                        std::to_string(i);
        auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
        last_op = rewriter.create<TileOp>(
            name_loc, newType, ValueRange{last_op}, attrs);
      }
    }
    return success();
  }
};

void RepeatOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<TopRepeatToTile>(context);
}

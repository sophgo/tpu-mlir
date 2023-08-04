//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/Patterns.h"

using namespace tpu_mlir::top;

struct ConvertExpand : public OpRewritePattern<ExpandOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExpandOp op,
                                PatternRewriter &rewriter) const override {
    auto output_shape = module::getShape(op.getOutput());
    auto input_shape = module::getShape(op.getInput());
    auto output_dims = output_shape.size();
    assert(output_dims >= input_shape.size());
    Value new_op = op.getInput();
    std::string name = module::getName(op.getOutput()).str();
    auto elt_type = module::getElementType(op.getOutput());

    rewriter.setInsertionPoint(op);
    // remove leading 1
    auto len_diff = output_dims - input_shape.size();
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    if (len_diff != 0) {
      std::vector<int64_t> input_shape_ex(len_diff, 1);
      for (auto s : input_shape) {
        input_shape_ex.push_back(s);
      }
      auto newType = RankedTensorType::get(input_shape_ex, elt_type);
      auto loc = NameLoc::get(rewriter.getStringAttr(name + "_reshape"));
      new_op = rewriter.create<ReshapeOp>(loc, newType, ValueRange{new_op});
      out_shape = input_shape_ex;
    }
    // tile one axis each time to avoid gmem buffer
    int32_t count = 0;
    for (uint32_t i = 0; i < output_dims; i++) {
      if (out_shape[i] != output_shape[i]) {
        ++count;
      }
    }
    if (count == 0) {
      op.replaceAllUsesWith(new_op.getDefiningOp());
      rewriter.eraseOp(op);
      return success();
    }
    for (uint32_t i = 1; i <= output_dims; i++) {
      int32_t axis = out_shape.size() - i;
      if (axis < 0) {
        out_shape.insert(out_shape.begin(), 1);
      }
      auto out_dims = out_shape.size();
      NameLoc loc;
      if (output_shape[output_dims - i] != out_shape[out_dims - i]) {
        auto tile = output_shape[output_dims - i] / out_shape[out_dims - i];
        if (count == 1) {
          loc = NameLoc::get(rewriter.getStringAttr(name));
          out_shape = output_shape;
        } else {
          loc = NameLoc::get(rewriter.getStringAttr(name + std::to_string(count)));
          out_shape[out_dims - i] = output_shape[output_dims - i];
        }
        auto newType = RankedTensorType::get(out_shape, elt_type);
        std::vector<NamedAttribute> attrs;
        attrs.push_back(
            rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(axis)));
        attrs.push_back(
            rewriter.getNamedAttr("tile", rewriter.getI64IntegerAttr(tile)));
        new_op =
            rewriter.create<TileOp>(loc, newType, ValueRange{new_op}, attrs);
        --count;
      }
      if (count == 0) {
        op.replaceAllUsesWith(new_op.getDefiningOp());
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
  }
};

void ExpandOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<ConvertExpand>(context);
}

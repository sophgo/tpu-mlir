//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace tpu_mlir::top;

struct TopPermuteToPixelShuffle : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto input_shape = module::getShape(op.getInput());
    if (input_shape.size() != 6) {
      return failure();
    }

    std::vector<int64_t> ps = {0, 1, 4, 2, 5, 3};
    auto order = module::getI64Array(op.getOrder());
    if (*order != ps) {
      return failure();
    }
    auto reshape_before =
        dyn_cast_or_null<ReshapeOp>(op.getInput().getDefiningOp());
    if (!reshape_before) {
      return failure();
    }
    auto nextOp = *op.getOutput().getUsers().begin();
    auto reshape_after = dyn_cast_or_null<ReshapeOp>(nextOp);
    if (!reshape_after) {
      return failure();
    }
    auto output_shape = module::getShape(reshape_after.getOutput());
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
    attrs.push_back(
        rewriter.getNamedAttr("block_h", rewriter.getI64IntegerAttr(upscale_factor)));
    attrs.push_back(
        rewriter.getNamedAttr("block_w", rewriter.getI64IntegerAttr(upscale_factor)));
    attrs.push_back(
        rewriter.getNamedAttr("in_is_NCHW", rewriter.getBoolAttr(true)));
    attrs.push_back(
        rewriter.getNamedAttr("out_is_NCHW", rewriter.getBoolAttr(true)));
    attrs.push_back(
        rewriter.getNamedAttr("swap_cr", rewriter.getBoolAttr(false)));
    rewriter.replaceOpWithNewOp<Depth2SpaceOp>(
        reshape_after, reshape_after.getResult().getType(),
        ValueRange{reshape_before.getInput()}, attrs);
    rewriter.eraseOp(op);
    rewriter.eraseOp(reshape_before);
    return success();
  }
};

// permute + permute or permute + reshape + permute
struct PermuteFuse : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto in = op.getInput();
    if (in.hasOneUse() == false) {
      return failure();
    }
    if (auto rop = dyn_cast<ReshapeOp>(in.getDefiningOp())) {
      in = rop.getInput();
      if (in.hasOneUse() == false) {
        return failure();
      }
    }
    auto permute_op = dyn_cast<PermuteOp>(in.getDefiningOp());
    if (!permute_op) {
      return failure();
    }
    // op order
    std::vector<int64_t> in0_shape = module::getShape(permute_op.getInput());
    auto in0_order = module::getI64Array(permute_op.getOrder());
    std::vector<int64_t> in1_shape = module::getShape(op.getInput());
    auto in1_order = module::getI64Array(op.getOrder());
    std::vector<int64_t> out1_shape = module::getShape(op.getOutput());
    std::vector<int64_t> in0_shape_fix;
    std::vector<int64_t> in0_order_fix;
    std::vector<int64_t> out0_shape_fix;
    std::vector<int64_t> in1_shape_fix;
    std::vector<int64_t> in1_order_fix;
    int to_dim;
    for (to_dim = 2; to_dim <= 5; to_dim++) {
      auto ret = permute_reset(in0_shape, *in0_order, in0_shape_fix,
                               in0_order_fix, to_dim);
      if (ret == false) {
        continue;
      }
      ret = permute_reset(in1_shape, *in1_order, in1_shape_fix, in1_order_fix,
                          to_dim);
      if (ret == false) {
        continue;
      }
      break;
    }
    if (to_dim > 5) {
      return failure();
    }
    for (auto o : in0_order_fix) {
      out0_shape_fix.push_back(in0_shape_fix[o]);
    }
    if (in1_shape_fix != out0_shape_fix) {
      return failure();
    }
    // test
    std::vector<int64_t> origin_data;
    for (int64_t i = 0; i < to_dim; i++) {
      origin_data.push_back(i);
    }
    std::vector<int64_t> result0_data;
    for (auto o : in0_order_fix) {
      result0_data.push_back(origin_data[o]);
    }
    std::vector<int64_t> result1_data;
    for (auto o : in1_order_fix) {
      result1_data.push_back(result0_data[o]);
    }
    if (result1_data != origin_data) {
      return failure();
    }
    // bingoo !
    if (out1_shape == in0_shape) {
      op.getOutput().replaceAllUsesWith(permute_op.getInput());
    } else {
      std::string in_name =
          module::getName(permute_op.getInput()).str() + "_Reshape";
      auto loc = NameLoc::get(rewriter.getStringAttr(in_name));
      rewriter.setInsertionPoint(op);
      auto rs_op = rewriter.create<ReshapeOp>(
          loc, op.getOutput().getType(), ValueRange{permute_op.getInput()});
      op.getOutput().replaceAllUsesWith(rs_op.getOutput());
    }
    return success();
  }
};

void PermuteOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<TopPermuteToPixelShuffle, PermuteFuse>(context);
}

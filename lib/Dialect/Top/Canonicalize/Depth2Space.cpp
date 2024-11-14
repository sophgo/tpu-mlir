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

/*
+ - - - - - - - - - - - - - - - - - - - -  +
' Pass:                                    '
'                                          '
' +---------+     +--------------------+   '     +----------------+
' |PermuteOp| --> |   Depth2SpaceOp    |   ' --> |  Depth2SpaceOp |
' +---------+     +--------------------+   '     +----------------+
'                                          '
+ - - - - - - - - - - - - - - - - - - - - -+
*/
struct Depth2SpaceWithPermuteOpt : public OpRewriterPatternEx<Depth2SpaceOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  Depth2SpaceWithPermuteOpt(mlir::MLIRContext *context)
      : OpRewriterPatternEx<Depth2SpaceOp>(context,
                                           "Depth2SpaceWithPermuteOpt") {}

  LogicalResult matchAndRewriteImpl(Depth2SpaceOp op,
                                    PatternRewriter &rewriter) const override {
    auto depth2space_input = op.getInput();
    auto permute_op =
        dyn_cast_or_null<PermuteOp>(depth2space_input.getDefiningOp());
    if (!permute_op) {
      return failure();
    }

    auto order = module::getI64Array(permute_op.getOrder());
    std::vector<int64_t> valid_permute_orders_NCHW_to_NHWC = {0, 2, 3, 1};
    std::vector<int64_t> valid_permute_orders_NHWC_to_NCHW = {0, 3, 1, 2};
    if (*order != valid_permute_orders_NCHW_to_NHWC &&
        *order != valid_permute_orders_NHWC_to_NCHW) {
      return failure();
    }

    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("is_CRD", rewriter.getBoolAttr(op.getIs_CRD())));
    attrs.push_back(rewriter.getNamedAttr(
        "is_inversed", rewriter.getBoolAttr(op.getIsInversed())));
    attrs.push_back(rewriter.getNamedAttr(
        "in_is_NCHW", rewriter.getBoolAttr(!op.getInIs_NCHW())));
    attrs.push_back(rewriter.getNamedAttr(
        "out_is_NCHW", rewriter.getBoolAttr(op.getOutIs_NCHW())));
    attrs.push_back(
        rewriter.getNamedAttr("swap_cr", rewriter.getBoolAttr(op.getSwapCr())));
    attrs.push_back(rewriter.getNamedAttr(
        "block_h", rewriter.getI64IntegerAttr(op.getBlockH())));
    attrs.push_back(rewriter.getNamedAttr(
        "block_w", rewriter.getI64IntegerAttr(op.getBlockW())));

    auto depth2space_output_type = op.getResult().getType();
    rewriter.replaceOpWithNewOp<Depth2SpaceOp>(
        op, depth2space_output_type, ValueRange{permute_op.getInput()}, attrs);
    rewriter.eraseOp(permute_op);
    return success();
  }
};

void Depth2SpaceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<Depth2SpaceWithPermuteOpt>(context);
}

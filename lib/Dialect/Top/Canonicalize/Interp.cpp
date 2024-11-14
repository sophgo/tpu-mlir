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

struct RemoveInterp : public OpRewriterPatternEx<InterpOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  RemoveInterp(mlir::MLIRContext *context)
      : OpRewriterPatternEx<InterpOp>(context, "RemoveInterp") {}

  LogicalResult matchAndRewriteImpl(InterpOp op,
                                    PatternRewriter &rewriter) const override {
    auto input_shape = module::getShape(op.getInput());
    auto output_shape = module::getShape(op.getOutput());
    auto scale_h = op.getScaleH().convertToDouble();
    auto scale_w = op.getScaleW().convertToDouble();
    if (scale_h == 1.0 && scale_w == 1.0) {
      op.getOutput().replaceAllUsesWith(op.getInput());
      rewriter.eraseOp(op);
      return success();
    }
    if ((float)output_shape[0] / input_shape[0] != 1 ||
        (float)output_shape[1] / input_shape[1] != 1) {
      llvm_unreachable("Interp now only support h/w");
    }
    return failure();
  }
};

struct InterpToUpsampleMergePattern : public OpRewriterPatternEx<InterpOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  InterpToUpsampleMergePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<InterpOp>(context, "InterpToUpsampleMergePattern") {
  }

  LogicalResult matchAndRewriteImpl(InterpOp op,
                                    PatternRewriter &rewriter) const override {
    if (!op->hasOneUse()) {
      return failure();
    }

    auto mode = op.getMode();
    std::string mode_name = mode.data();
    if (mode_name != "nearest")
      return failure();

    auto input_shape = module::getShape(op.getInput());
    auto output_shape = module::getShape(op.getOutput());
    auto scale_h = op.getScaleH().convertToDouble();
    auto scale_w = op.getScaleW().convertToDouble();

    if (output_shape.size() <= 3 || output_shape[2] % input_shape[2] != 0 ||
        (double)output_shape[2] / (double)input_shape[2] !=
            (double)output_shape[3] / (double)input_shape[3]) {
      return failure();
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "scale_h", rewriter.getI64IntegerAttr((int64_t)scale_h)));
    attrs.push_back(rewriter.getNamedAttr(
        "scale_w", rewriter.getI64IntegerAttr((int64_t)scale_w)));

    rewriter.replaceOpWithNewOp<UpsampleOp>(op, op.getResult().getType(),
                                            op.getInput(), attrs);
    return success();
  }
};

void InterpOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<RemoveInterp, InterpToUpsampleMergePattern>(context);
}

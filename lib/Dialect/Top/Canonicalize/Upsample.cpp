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

struct RemoveUpsample : public OpRewriterPatternEx<UpsampleOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  RemoveUpsample(mlir::MLIRContext *context)
      : OpRewriterPatternEx<UpsampleOp>(context, "RemoveUpsample") {}

  LogicalResult matchAndRewriteImpl(UpsampleOp op,
                                    PatternRewriter &rewriter) const override {
    auto input_shape = module::getShape(op.getInput());
    auto output_shape = module::getShape(op.getOutput());
    auto scale_h = op.getScaleH();
    auto scale_w = op.getScaleW();
    if (scale_h == 1 && scale_w == 1) {
      op.getOutput().replaceAllUsesWith(op.getInput());
      rewriter.eraseOp(op);
      return success();
    }
    if (input_shape.size() != 4) {
      llvm_unreachable("Todo, support interp other then 4dims");
    } else if ((float)output_shape[0] / input_shape[0] != 1 ||
               (float)output_shape[1] / input_shape[1] != 1) {
      llvm_unreachable("Interp now only support h/w");
    }
    return failure();
  }
};

void UpsampleOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<RemoveUpsample>(context);
}

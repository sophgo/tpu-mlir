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
#include <set>

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct InterpToUpsampleMergePattern : public OpRewritePattern<InterpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InterpOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasOneUse()) {
      return failure();
    }

    auto mode = op.getMode();
    std::string mode_name = mode.data();
    if(mode_name != "nearest")   return failure();

    auto input_shape = module::getShape(op.getInput());
    auto output_shape = module::getShape(op.getOutput());
    auto scale_h = op.getScaleH().convertToDouble();
    auto scale_w = op.getScaleW().convertToDouble();
    
    if(output_shape[2] % input_shape[2]!=0 || output_shape[3] % input_shape[3]!=0){
        return failure();
    }
    int size = output_shape[2] / input_shape[2];
    if(size != output_shape[3] / input_shape[3]) return failure();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("scale_h", rewriter.getI64IntegerAttr((int64_t)scale_h)));
    attrs.push_back(
        rewriter.getNamedAttr("scale_w", rewriter.getI64IntegerAttr((int64_t)scale_w)));

    rewriter.replaceOpWithNewOp<UpsampleOp>(op, op.getResult().getType(),
                                           op.getInput(), attrs);
    return success();
  }
};

void InterpOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<InterpToUpsampleMergePattern>(context);
}
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace tpu_mlir::top;

struct Deconv1dTo2d : public OpRewritePattern<DeconvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeconvOp op,
                                PatternRewriter &rewriter) const override {

    auto kernel = module::getI64Array(op.getKernelShape());
    if (kernel->size() != 1) {
      return failure();
    }
    std::vector<int64_t> vfilterShape = module::getShape(op.getFilter());
    vfilterShape.push_back(1);
    auto new_type = RankedTensorType::get(vfilterShape, rewriter.getF32Type());
    op.getFilter().setType(new_type);
    // update kernel_shape
    std::vector<int64_t> kernel_shape =
        *module::getI64Array(op.getKernelShape());
    kernel_shape.push_back(1);
    op.setKernelShapeAttr(rewriter.getI64ArrayAttr(kernel_shape));
    std::vector<int64_t> strides = *module::getI64Array(op.getStrides());
    strides.push_back(1);
    op.setStridesAttr(rewriter.getI64ArrayAttr(strides));
    // update pads
    auto pads_v = module::getI64Array(op.getPads());
    std::vector<int64_t> pads = {pads_v->at(0), 0, pads_v->at(1), 0};
    op.setPadsAttr(rewriter.getI64ArrayAttr(pads));
    // update dilations
    std::vector<int64_t> dilations =
        *module::getI64Array(op.getDilations(), 1, 1);
    dilations.push_back(1);
    op.setDilationsAttr(rewriter.getI64ArrayAttr(dilations));
    // update output_padding
    std::vector<int64_t> output_padding =
        *module::getI64Array(op.getOutputPadding(), 1, 0);
    output_padding.push_back(0);
    op.setOutputPaddingAttr(rewriter.getI64ArrayAttr(output_padding));
    return success();
  }
};

void DeconvOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<Deconv1dTo2d>(context);
}

//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::top;

struct TopBatchNormToDwConv : public OpRewritePattern<BatchNormOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchNormOp op,
                                PatternRewriter &rewriter) const override {

    auto mean = dyn_cast<top::WeightOp>(op.mean().getDefiningOp());
    auto variance = dyn_cast<top::WeightOp>(op.variance().getDefiningOp());
    auto gamma = dyn_cast<top::WeightOp>(op.gamma().getDefiningOp());
    auto beta = dyn_cast<top::WeightOp>(op.beta().getDefiningOp());
    if (!(mean && variance && gamma && beta))
      return failure();
    auto mean_d = mean.read<float>();
    auto variance_d = variance.read<float>();
    auto gamma_d = gamma.read<float>();
    auto beta_d = beta.read<float>();

    int channel = gamma.getType().cast<RankedTensorType>().getNumElements();
    std::vector<float> scale(channel);
    std::vector<float> bias(channel);

    auto eps = op.epsilon().convertToDouble();
    for (int i = 0; i < channel; ++i) {
      scale[i] = 1 / std::sqrt(variance_d->at(i) + eps) * gamma_d->at(i);
      bias[i] = -mean_d->at(i) * scale[i] + beta_d->at(i);
    }
    // batch normal to depthwise convolution
    NamedAttrList attrs;

    attrs.set("kernel_shape", rewriter.getI64ArrayAttr({1, 1}));
    attrs.set("strides", rewriter.getI64ArrayAttr({1, 1}));
    attrs.set("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0}));
    attrs.set("group", rewriter.getI64IntegerAttr(channel));
    attrs.set("do_relu", rewriter.getBoolAttr(op.do_relu()));

    auto filter_type =
        RankedTensorType::get({channel, 1, 1, 1}, rewriter.getF32Type());
    auto new_scale =
        top::WeightOp::create(mean, "merged_to_scale", scale, filter_type);
    auto bias_type = RankedTensorType::get({channel}, rewriter.getF32Type());
    auto new_bias =
        top::WeightOp::create(mean, "merged_to_bias", bias, bias_type);

    rewriter.replaceOpWithNewOp<ConvOp>(
        op, op.getResult().getType(),
        ValueRange{op.input(), new_scale, new_bias}, attrs);
    return success();
  }
};

void BatchNormOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<TopBatchNormToDwConv>(context);
}

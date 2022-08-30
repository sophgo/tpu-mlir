//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
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

struct TopBatchNormMergeToConv : public OpRewritePattern<BatchNormOp> {
  using OpRewritePattern::OpRewritePattern;
  TopBatchNormMergeToConv(MLIRContext *context, PatternBenefit benefit = 9)
      : OpRewritePattern<BatchNormOp>(context, benefit) {}

  LogicalResult matchAndRewrite(BatchNormOp op,
                                PatternRewriter &rewriter) const override {
    auto formerOp = op.input().getDefiningOp();
    if (!formerOp->getResult(0).hasOneUse() || !isa<ConvOp>(formerOp)) {
      return failure();
    }
    auto conv_op = cast<ConvOp>(formerOp);
    if (conv_op.do_relu()) {
      return failure();
    }

    auto gamma = dyn_cast<WeightOp>(op.gamma().getDefiningOp());
    auto beta = dyn_cast<WeightOp>(op.beta().getDefiningOp());
    auto mean = dyn_cast<WeightOp>(op.mean().getDefiningOp());
    auto variance = dyn_cast<WeightOp>(op.variance().getDefiningOp());
    if (!(mean && variance && gamma && beta))
      return failure();
    auto gamma_f32 = gamma.read<float>();
    auto beta_f32 = beta.read<float>();
    auto mean_f32 = mean.read<float>();
    auto variance_f32 = variance.read<float>();

    auto conv_weight_op = dyn_cast<WeightOp>(conv_op.filter().getDefiningOp());
    auto conv_bias_op = dyn_cast<WeightOp>(conv_op.bias().getDefiningOp());

    int64_t oc, ic, kh, kw;
    Module::getNCHW(conv_weight_op.output(), oc, ic, kh, kw);

    // int channel = gamma.getType().cast<RankedTensorType>().getNumElements();
    std::vector<float> scale(oc);
    std::vector<float> bias(oc);

    // constructe scale and bias by params of BatchNorm
    auto eps = op.epsilon().convertToDouble();
    for (int i = 0; i < oc; ++i) {
      scale[i] = 1 / std::sqrt(variance_f32->at(i) + eps) * gamma_f32->at(i);
      bias[i] = -mean_f32->at(i) * scale[i] + beta_f32->at(i);
    }

    // merge weight: weight = weight * scale
    std::vector<float> conv_weight_v(oc * ic * kh * kw, 0);
    auto conv_weight_f32 = conv_weight_op.read<float>();
    for (int i = 0; i < oc; ++i) {
      for (int j = 0; j < kw * kh * ic; ++j) {
        conv_weight_v[i * ic * kh * kw + j] =
            conv_weight_f32->at(i * ic * kh * kw + j) * scale[i];
      }
    }
    // merge bias: bias = bias * scale + bias
    std::vector<float> conv_bias_v(oc, 0);
    if (conv_bias_op != nullptr) {
      auto conv_bias_f32 = conv_bias_op.read<float>();
      for (int i = 0; i < oc; ++i) {
        conv_bias_v[i] = conv_bias_f32->at(i) * scale[i] + bias[i];
      }
    } else {
      for (int i = 0; i < oc; ++i) {
        conv_bias_v[i] = bias[i];
      }
    }

    auto weight_type =
        RankedTensorType::get({oc, ic, kh, kw}, rewriter.getF32Type());
    auto conv_weight = WeightOp::create(conv_op, "bn_merged_to_conv_weight",
                                        conv_weight_v, weight_type);
    auto bias_type = RankedTensorType::get({oc}, rewriter.getF32Type());
    auto conv_bias = WeightOp::create(conv_op, "bn_merged_to_conv_bias",
                                      conv_bias_v, bias_type);
    conv_op->setOperand(1, conv_weight);
    conv_op->setOperand(2, conv_bias);

    // update attrs
    double relu_limit = op.relu_limit().convertToDouble();
    formerOp->setAttr("do_relu", rewriter.getBoolAttr(op.do_relu()));
    formerOp->setAttr("relu_limit", rewriter.getF64FloatAttr(relu_limit));

    // remove the scale Op
    rewriter.replaceOp(op, {op.input()});
    return success();
  }
};

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
        WeightOp::create(mean, "merged_to_scale", scale, filter_type);
    auto bias_type = RankedTensorType::get({channel}, rewriter.getF32Type());
    auto new_bias =
        WeightOp::create(mean, "merged_to_bias", bias, bias_type);

    rewriter.replaceOpWithNewOp<ConvOp>(
        op, op.getResult().getType(),
        ValueRange{op.input(), new_scale, new_bias}, attrs);
    return success();
  }
};

void BatchNormOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<TopBatchNormMergeToConv, TopBatchNormToDwConv>(context);
}

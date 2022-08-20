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

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;
using namespace tpu_mlir::helper;

struct TopMultiScaleMergeToOne : public OpRewritePattern<ScaleOp> {
  using OpRewritePattern::OpRewritePattern;
  TopMultiScaleMergeToOne(MLIRContext *context, PatternBenefit benefit = 10)
      : OpRewritePattern<ScaleOp>(context, benefit) {}

  LogicalResult matchAndRewrite(ScaleOp op,
                                PatternRewriter &rewriter) const override {
    auto nextOp = *op->getUsers().begin();
    if (!op->getResult(0).hasOneUse() || !isa<ScaleOp>(nextOp)) {
      return failure();
    }

    auto next_scale_op = cast<ScaleOp>(nextOp);
    auto next_scale = dyn_cast<WeightOp>(next_scale_op.scale().getDefiningOp());
    auto next_bias = dyn_cast<WeightOp>(next_scale_op.bias().getDefiningOp());
    auto next_scale_f32 = next_scale.read<float>();
    auto next_bias_f32 = next_bias.read<float>();

    auto cur_scale = dyn_cast<WeightOp>(op.scale().getDefiningOp());
    auto cur_bias = dyn_cast<WeightOp>(op.bias().getDefiningOp());
    auto cur_scale_f32 = cur_scale.read<float>();
    auto cur_bias_f32 = cur_bias.read<float>();

    int channel = cur_scale.getType().cast<RankedTensorType>().getNumElements();
    std::vector<float> scale_v(channel);
    std::vector<float> bias_v(channel);
    for (int i = 0; i < channel; ++i) {
      scale_v[i] = cur_scale_f32->at(i) * next_scale_f32->at(i);
      bias_v[i] =
          cur_bias_f32->at(i) * next_scale_f32->at(i) + next_bias_f32->at(i);
    }

    auto scale_type = RankedTensorType::get({channel}, rewriter.getF32Type());
    auto new_scale =
        WeightOp::create(nextOp, "merged_scale", scale_v, scale_type);
    auto bias_type = RankedTensorType::get({channel}, rewriter.getF32Type());
    auto new_bias = WeightOp::create(nextOp, "merged_bias", bias_v, bias_type);
    nextOp->setOperand(1, new_scale);
    nextOp->setOperand(2, new_bias);

    rewriter.replaceOp(op, {op.input()});
    return success();
  }
};

struct TopScaleMergeToConv : public OpRewritePattern<ScaleOp> {
  using OpRewritePattern::OpRewritePattern;
  TopScaleMergeToConv(MLIRContext *context, PatternBenefit benefit = 9)
      : OpRewritePattern<ScaleOp>(context, benefit) {}

  LogicalResult matchAndRewrite(ScaleOp op,
                                PatternRewriter &rewriter) const override {
    auto formerOp = op.input().getDefiningOp();
    if (!formerOp->getResult(0).hasOneUse() || !isa<ConvOp>(formerOp)) {
      return failure();
    }
    auto conv_op = cast<ConvOp>(formerOp);
    if (conv_op.do_relu()) {
      return failure();
    }

    auto cur_scale_op = dyn_cast<WeightOp>(op.scale().getDefiningOp());
    auto cur_bias_op = dyn_cast<WeightOp>(op.bias().getDefiningOp());
    auto cur_scale_f32 = cur_scale_op.read<float>();
    auto cur_bias_f32 = cur_bias_op.read<float>();

    auto conv_weight_op = dyn_cast<WeightOp>(conv_op.filter().getDefiningOp());
    auto conv_bias_op = dyn_cast<WeightOp>(conv_op.bias().getDefiningOp());

    int64_t oc, ic, kh, kw;
    Module::getNCHW(conv_weight_op.output(), oc, ic, kh, kw);

    // merge weight: weight = weight * cur_scale
    std::vector<float> conv_weight_v(oc * ic * kh * kw, 0);
    auto conv_weight_f32 = conv_weight_op.read<float>();
    for (int i = 0; i < oc; ++i) {
      for (int j = 0; j < kw * kh * ic; ++j) {
        conv_weight_v[i * ic * kh * kw + j] =
            conv_weight_f32->at(i * ic * kh * kw + j) * cur_scale_f32->at(i);
      }
    }
    // merge bias: bias = bias * cur_scale + cur_bias
    std::vector<float> conv_bias_v(oc, 0);
    if (conv_bias_op != nullptr) {
      auto conv_bias_f32 = conv_bias_op.read<float>();
      for (int i = 0; i < oc; ++i) {
        conv_bias_v[i] =
            conv_bias_f32->at(i) * cur_scale_f32->at(i) + cur_bias_f32->at(i);
      }
    } else {
      for (int i = 0; i < oc; ++i) {
        conv_bias_v[i] = cur_bias_f32->at(i);
      }
    }

    auto weight_type =
        RankedTensorType::get({oc * ic * kh * kw}, rewriter.getF32Type());
    auto conv_weight = WeightOp::create(conv_op, "merged_to_conv_weight",
                                        conv_weight_v, weight_type);
    auto bias_type = RankedTensorType::get({oc}, rewriter.getF32Type());
    auto conv_bias = WeightOp::create(conv_op, "merged_to_conv_bias",
                                      conv_bias_v, bias_type);
    conv_op->setOperand(1, conv_weight);
    conv_op->setOperand(2, conv_bias);

    // update attrs
    double relu_limit = op.relu_limit().convertToDouble();
    formerOp->setAttr("do_relu", rewriter.getBoolAttr(op.do_relu()));
    formerOp->setAttr("relu_limit", rewriter.getF64FloatAttr(relu_limit));
    formerOp->setAttr("name", op.nameAttr());

    // remove the scale Op
    rewriter.replaceOp(op, {op.input()});
    return success();
  }
};

struct TopScaleMergeToBatchNorm : public OpRewritePattern<ScaleOp> {
  using OpRewritePattern::OpRewritePattern;
  TopScaleMergeToBatchNorm(MLIRContext *context, PatternBenefit benefit = 9)
      : OpRewritePattern<ScaleOp>(context, benefit) {}

  LogicalResult matchAndRewrite(ScaleOp op,
                                PatternRewriter &rewriter) const override {
    auto formerOp = op.input().getDefiningOp();
    if (!formerOp->getResult(0).hasOneUse() || !isa<BatchNormOp>(formerOp)) {
      return failure();
    }
    auto bn_op = cast<BatchNormOp>(formerOp);
    if (bn_op.do_relu()) {
      return failure();
    }

    auto cur_scale_op = dyn_cast<WeightOp>(op.scale().getDefiningOp());
    auto cur_bias_op = dyn_cast<WeightOp>(op.bias().getDefiningOp());
    auto cur_scale_f32 = cur_scale_op.read<float>();
    auto cur_bias_f32 = cur_bias_op.read<float>();

    auto bn_mean_op = dyn_cast<WeightOp>(bn_op.mean().getDefiningOp());
    auto bn_variance_op = dyn_cast<WeightOp>(bn_op.variance().getDefiningOp());
    auto bn_mean_f32 = bn_mean_op.read<float>();
    auto bn_variance_f32 = bn_variance_op.read<float>();

    int channel =
        bn_mean_op.getType().cast<RankedTensorType>().getNumElements();
    std::vector<float> bn_mean_v(channel, 0);
    std::vector<float> bn_variance_v(channel, 0);
    // update weight
    for (int i = 0; i < channel; ++i) {
      float divisor = cur_scale_f32->at(i) * bn_variance_f32->at(i);
      if (std::fabs(divisor) > 1e-6) {
        bn_mean_v[i] = bn_mean_f32->at(i) + cur_bias_f32->at(i) / divisor;
      } else {
        bn_mean_v[i] = bn_mean_f32->at(i) +
                       cur_bias_f32->at(i) * (divisor >= 0 ? 1e20 : -1e20);
      }
    }
    // update variance:
    for (int i = 0; i < channel; ++i) {
      float divisor = cur_scale_f32->at(i) * bn_variance_f32->at(i);
      if (std::fabs(divisor) > 1e-6) {
        bn_variance_v[i] = divisor;
      } else {
        bn_variance_v[i] = 1. / (divisor >= 0 ? 1e20 : -1e20);
      }
    }

    auto mean_type = RankedTensorType::get({channel}, rewriter.getF32Type());
    auto bn_mean =
        WeightOp::create(bn_op, "merged_to_bn_mean", bn_mean_v, mean_type);
    auto variance_type =
        RankedTensorType::get({channel}, rewriter.getF32Type());
    auto bn_variance = WeightOp::create(bn_op, "merged_to_bn_variance",
                                        bn_variance_v, variance_type);
    bn_op->setOperand(1, bn_mean);
    bn_op->setOperand(2, bn_variance);

    // update attrs
    double relu_limit = op.relu_limit().convertToDouble();
    formerOp->setAttr("do_relu", rewriter.getBoolAttr(op.do_relu()));
    formerOp->setAttr("relu_limit", rewriter.getF64FloatAttr(relu_limit));
    formerOp->setAttr("name", op.nameAttr());

    // remove the scale Op
    rewriter.replaceOp(op, {op.input()});
    return success();
  }
};

struct TopScaleToDwConv : public OpRewritePattern<ScaleOp> {
  using OpRewritePattern::OpRewritePattern;
  TopScaleToDwConv(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<ScaleOp>(context, benefit) {}

  LogicalResult matchAndRewrite(ScaleOp op,
                                PatternRewriter &rewriter) const override {
    auto cur_scale = dyn_cast<WeightOp>(op.scale().getDefiningOp());
    auto cur_bias = dyn_cast<WeightOp>(op.bias().getDefiningOp());
    if (!(cur_scale && cur_bias)) {
      return failure();
    }
    int channel = cur_scale.getType().cast<RankedTensorType>().getNumElements();
    auto cur_scale_f32 = cur_scale.read<float>();
    auto cur_bias_f32 = cur_bias.read<float>();

    std::vector<float> new_scale_v(channel);
    std::vector<float> new_bias_v(channel);
    std::copy(cur_scale_f32->begin(), cur_scale_f32->end(),
              new_scale_v.begin());
    std::copy(cur_bias_f32->begin(), cur_bias_f32->end(), new_bias_v.begin());

    // scale to depthwise convolution
    NamedAttrList attrs;
    attrs.set("name", op.nameAttr());
    attrs.set("kernel_shape", rewriter.getI64ArrayAttr({1, 1}));
    attrs.set("strides", rewriter.getI64ArrayAttr({1, 1}));
    attrs.set("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0}));
    attrs.set("group", rewriter.getI64IntegerAttr(channel));
    attrs.set("do_relu", rewriter.getBoolAttr(op.do_relu()));
    auto relu_limit = op.relu_limit().convertToDouble();
    attrs.set("relu_limit", rewriter.getF64FloatAttr(relu_limit));

    auto filter_type =
        RankedTensorType::get({channel, 1, 1, 1}, rewriter.getF32Type());
    auto new_scale =
        WeightOp::create(op, "merged_to_weight", new_scale_v, filter_type);
    auto bias_type = RankedTensorType::get({channel}, rewriter.getF32Type());
    auto new_bias =
        WeightOp::create(op, "merged_to_bias", new_bias_v, bias_type);

    rewriter.replaceOpWithNewOp<ConvOp>(
        op, op.getResult().getType(),
        ValueRange{op.input(), new_scale, new_bias}, attrs);
    return success();
  }
};

void ScaleOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<TopScaleToDwConv, TopScaleMergeToConv, TopMultiScaleMergeToOne,
                 TopScaleMergeToBatchNorm>(context);
}

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct TopMultiScaleMergeToOne : public OpRewritePattern<ScaleOp> {
  using OpRewritePattern::OpRewritePattern;
  TopMultiScaleMergeToOne(MLIRContext *context, PatternBenefit benefit = 10)
      : OpRewritePattern<ScaleOp>(context, benefit) {}

  LogicalResult matchAndRewrite(ScaleOp op,
                                PatternRewriter &rewriter) const override {
    auto nextOp = *op->getUsers().begin();
    if (!op->hasOneUse() || !isa<ScaleOp>(nextOp)) {
      return failure();
    }

    auto next_scale_op = cast<ScaleOp>(nextOp);
    auto next_scale =
        dyn_cast<WeightOp>(next_scale_op.getScale().getDefiningOp());
    auto next_bias =
        dyn_cast<WeightOp>(next_scale_op.getBias().getDefiningOp());
    auto next_scale_f32 = next_scale.read<float>();
    auto next_bias_f32 = next_bias.read<float>();

    auto cur_scale = dyn_cast<WeightOp>(op.getScale().getDefiningOp());
    auto cur_bias = dyn_cast<WeightOp>(op.getBias().getDefiningOp());
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

    rewriter.replaceOp(op, {op.getInput()});
    return success();
  }
};

struct TopScaleMergeToConv : public OpRewritePattern<ScaleOp> {
  using OpRewritePattern::OpRewritePattern;
  TopScaleMergeToConv(MLIRContext *context, PatternBenefit benefit = 9)
      : OpRewritePattern<ScaleOp>(context, benefit) {}

  LogicalResult matchAndRewrite(ScaleOp op,
                                PatternRewriter &rewriter) const override {
    auto formerOp = op.getInput().getDefiningOp();
    if (!formerOp->hasOneUse() || !isa<ConvOp>(formerOp)) {
      return failure();
    }
    auto conv_op = cast<ConvOp>(formerOp);
    if (conv_op.getDoRelu()) {
      return failure();
    }

    auto cur_scale_op = dyn_cast<WeightOp>(op.getScale().getDefiningOp());
    auto cur_bias_op = dyn_cast<WeightOp>(op.getBias().getDefiningOp());
    auto cur_scale_f32 = cur_scale_op.read<float>();
    auto cur_bias_f32 = cur_bias_op.read<float>();

    auto conv_weight_op =
        dyn_cast<WeightOp>(conv_op.getFilter().getDefiningOp());
    auto conv_bias_op = dyn_cast<WeightOp>(conv_op.getBias().getDefiningOp());

    int64_t oc, ic, kh, kw;
    module::getNCHW(conv_weight_op.getOutput(), oc, ic, kh, kw);

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
        RankedTensorType::get({oc, ic, kh, kw}, rewriter.getF32Type());
    auto conv_weight = WeightOp::create(conv_op, "merged_to_conv_weight",
                                        conv_weight_v, weight_type);
    auto bias_type = RankedTensorType::get({oc}, rewriter.getF32Type());
    auto conv_bias = WeightOp::create(conv_op, "merged_to_conv_bias",
                                      conv_bias_v, bias_type);
    conv_op->setOperand(1, conv_weight);
    conv_op->setOperand(2, conv_bias);

    // update attrs
    double relu_limit = op.getReluLimit().convertToDouble();
    formerOp->setLoc(op.getLoc());
    formerOp->setAttr("do_relu", rewriter.getBoolAttr(op.getDoRelu()));
    formerOp->setAttr("relu_limit", rewriter.getF64FloatAttr(relu_limit));

    // remove the scale Op
    rewriter.replaceOp(op, {op.getInput()});
    return success();
  }
};

struct ConstbinaryMergeToTopScale : public OpRewritePattern<ScaleOp> {
  using OpRewritePattern::OpRewritePattern;
  ConstbinaryMergeToTopScale(MLIRContext *context, PatternBenefit benefit = 6)
      : OpRewritePattern<ScaleOp>(context, benefit) {}

  LogicalResult matchAndRewrite(ScaleOp op,
                                PatternRewriter &rewriter) const override {
    auto formerOp = op->getOperand(0).getDefiningOp();

    if (!isa<MulConstOp, AddConstOp>(formerOp)) {
      return failure();
    }

    auto scale = dyn_cast<WeightOp>(op.getScale().getDefiningOp());
    auto bias = dyn_cast<WeightOp>(op.getBias().getDefiningOp());
    auto scale_f32 = scale.read<float>();
    auto bias_f32 = bias.read<float>();

    int elem_num = scale.getType().cast<RankedTensorType>().getNumElements();
    std::vector<float> scale_v(elem_num);
    std::vector<float> bias_v(elem_num);

    if (isa<MulConstOp>(formerOp)) {
      auto mul_const_op = cast<MulConstOp>(formerOp);
      float value = mul_const_op.getConstVal().convertToFloat();
      for (int i = 0; i < elem_num; ++i) {
        scale_v[i] = scale_f32->at(i) * value;
      }
      auto scale_type =
          RankedTensorType::get({elem_num}, rewriter.getF32Type());
      auto new_scale = WeightOp::create(op, "constbinary_merged_to_scale",
                                        scale_v, scale_type);
      op->setOperand(1, new_scale);
      op->setOperand(2, bias);
      rewriter.replaceOp(mul_const_op, {op});
      return success();
    }

    if (isa<AddConstOp>(formerOp)) {
      auto add_const_op = cast<AddConstOp>(formerOp);
      float value = add_const_op.getConstVal().convertToFloat();
      for (int i = 0; i < elem_num; ++i) {
        bias_v[i] += scale_f32->at(i) * value;
      }
      auto bias_type = RankedTensorType::get({elem_num}, rewriter.getF32Type());
      auto new_bias =
          WeightOp::create(op, "constbinary_merged_to_bias", bias_v, bias_type);
      op->setOperand(1, scale);
      op->setOperand(2, new_bias);
      rewriter.replaceOp(add_const_op, {op});
      return success();
    }
    return failure();
  }
};

struct TopScaleMergeToBatchNorm : public OpRewritePattern<ScaleOp> {
  using OpRewritePattern::OpRewritePattern;
  TopScaleMergeToBatchNorm(MLIRContext *context, PatternBenefit benefit = 9)
      : OpRewritePattern<ScaleOp>(context, benefit) {}

  LogicalResult matchAndRewrite(ScaleOp op,
                                PatternRewriter &rewriter) const override {
    auto formerOp = op.getInput().getDefiningOp();
    if (!formerOp->getResult(0).hasOneUse() || !isa<BatchNormOp>(formerOp)) {
      return failure();
    }
    auto bn_op = cast<BatchNormOp>(formerOp);
    if (bn_op.getDoRelu()) {
      return failure();
    }

    auto cur_scale_op = dyn_cast<WeightOp>(op.getScale().getDefiningOp());
    auto cur_bias_op = dyn_cast<WeightOp>(op.getBias().getDefiningOp());
    auto cur_scale_f32 = cur_scale_op.read<float>();
    auto cur_bias_f32 = cur_bias_op.read<float>();

    auto bn_mean_op = dyn_cast<WeightOp>(bn_op.getMean().getDefiningOp());
    auto bn_variance_op =
        dyn_cast<WeightOp>(bn_op.getVariance().getDefiningOp());
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
    double relu_limit = op.getReluLimit().convertToDouble();
    formerOp->setAttr("do_relu", rewriter.getBoolAttr(op.getDoRelu()));
    formerOp->setAttr("relu_limit", rewriter.getF64FloatAttr(relu_limit));

    // remove the scale Op
    rewriter.replaceOp(op, {op.getInput()});
    return success();
  }
};
struct ScaleShapeAlign : public OpRewritePattern<ScaleOp> {
  using OpRewritePattern::OpRewritePattern;
  ScaleShapeAlign(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<ScaleOp>(context, benefit) {}

  LogicalResult matchAndRewrite(ScaleOp op,
                                PatternRewriter &rewriter) const override {
    auto input_shape = module::getShape(op.getInput());
    bool changed = false;
    for (auto operand : op->getOperands()) {
      if (auto weight = dyn_cast<WeightOp>(op.getScale().getDefiningOp())) {
        auto weight_shape = module::getShape(operand);
        if (weight_shape.size() == 1 && input_shape[1] == weight_shape[0]) {
          auto expand_shape = RankedTensorType::get(
              {1, weight_shape[0]}, module::getElementType(operand));
          operand.setType(expand_shape);
          changed = true;
        }
      }
    }
    if (changed)
      return success();
    else
      return failure();
  }
};

struct TopScaleMergeToMatMul : public OpRewritePattern<ScaleOp> {
  using OpRewritePattern::OpRewritePattern;
  TopScaleMergeToMatMul(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<ScaleOp>(context, benefit) {}
  LogicalResult matchAndRewrite(ScaleOp op,
                                PatternRewriter &rewriter) const override {
    auto preOp = op.getInput().getDefiningOp();
    if (!preOp->hasOneUse() || !isa<MatMulOp>(preOp)) {
      return failure();
    }
    auto matmulOp = cast<MatMulOp>(preOp);
    auto weight = dyn_cast<WeightOp>(matmulOp.getRight().getDefiningOp());
    auto scale = dyn_cast<WeightOp>(op.getScale().getDefiningOp());
    auto bias = dyn_cast<WeightOp>(op.getBias().getDefiningOp());
    auto input_shape = module::getShape(op.getInput());
    if (!weight || !scale || !bias || input_shape.size() != 2) {
      return failure();
    }

    // merge scale into matmul's right weight
    auto weight_data = weight.read<float>();
    auto scale_data = scale.read<float>();
    auto right_shape = module::getShape(matmulOp.getRight());
    auto N = scale.getType().cast<RankedTensorType>().getNumElements();
    assert(right_shape[1] == N);
    for (int k = 0; k < right_shape[0]; ++k) {
      for (int n = 0; n < right_shape[1]; ++n) {
        weight_data->at(k * right_shape[1] + n) *= scale_data->at(n);
      }
    }
    auto weight_type = RankedTensorType::get({right_shape[0], right_shape[1]},
                                             rewriter.getF32Type());
    auto new_weight = WeightOp::create(matmulOp, "merged_scale_to_matmul",
                                       *weight_data, weight_type);
    matmulOp.setOperand(1, new_weight);

    // merge bias into matmul's bias
    auto bias_data = bias.read<float>();
    std::vector<float> new_bias_v(N, 0);
    new_bias_v.assign(bias_data->begin(), bias_data->end());
    auto bias_type = RankedTensorType::get({N}, rewriter.getF32Type());
    if (!module::isNone(matmulOp.getBias())) {
      auto matmul_bias_data =
          dyn_cast<WeightOp>(matmulOp.getBias().getDefiningOp()).read<float>();
      for (int n = 0; n < N; ++n) {
        new_bias_v[n] += matmul_bias_data->at(n) * scale_data->at(n);
      }
    }
    bool bias_all_zeros = std::all_of(new_bias_v.begin(), new_bias_v.end(),
                                      [](float i) { return i == 0.f; });
    if (!bias_all_zeros || !module::isNone(matmulOp.getBias())) {
      auto new_bias = WeightOp::create(matmulOp, "merged_bias_to_matmul",
                                       new_bias_v, bias_type);
      matmulOp.setOperand(2, new_bias);
    }

    // update attrs
    preOp->setLoc(op.getLoc());
    preOp->setAttr("do_relu", rewriter.getBoolAttr(op.getDoRelu()));
    preOp->setAttr("relu_limit", rewriter.getF64FloatAttr(
                                     op.getReluLimit().convertToDouble()));
    // remove scale Op
    rewriter.replaceOp(op, {op.getInput()});
    return success();
  }
};

void ScaleOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<TopScaleMergeToConv, TopMultiScaleMergeToOne,
                 TopScaleMergeToBatchNorm, ScaleShapeAlign,
                 ConstbinaryMergeToTopScale, TopScaleMergeToMatMul>(context);
}

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

struct TopMultiScaleMergeToOne : public OpRewriterPatternEx<ScaleOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  TopMultiScaleMergeToOne(MLIRContext *context, PatternBenefit benefit = 10)
      : OpRewriterPatternEx<ScaleOp>(context, "TopMultiScaleMergeToOne",
                                     benefit) {}

  LogicalResult matchAndRewriteImpl(ScaleOp op,
                                    PatternRewriter &rewriter) const override {

    if (op.getDoRelu()) {
      return failure();
    }
    auto nextOp = *op->getUsers().begin();
    if (!op->hasOneUse() || !isa<ScaleOp>(nextOp)) {
      return failure();
    }

    auto cur_storage_type = module::getStorageType(op.getOutput());
    if (!cur_storage_type.isF32() && !cur_storage_type.isF16()) {
      return failure();
    }

    auto next_scale_op = cast<ScaleOp>(nextOp);
    auto next_storage_type = module::getStorageType(next_scale_op.getOutput());
    if (cur_storage_type != next_storage_type) {
      return failure();
    }

    auto next_scale =
        dyn_cast<WeightOp>(next_scale_op.getScale().getDefiningOp());
    auto next_bias =
        dyn_cast<WeightOp>(next_scale_op.getBias().getDefiningOp());
    auto next_scale_f32 = next_scale.read_as_float();
    auto next_bias_f32 = next_bias.read_as_float();

    auto cur_scale = dyn_cast<WeightOp>(op.getScale().getDefiningOp());
    auto cur_bias = dyn_cast<WeightOp>(op.getBias().getDefiningOp());
    auto cur_scale_f32 = cur_scale.read_as_float();
    auto cur_bias_f32 = cur_bias.read_as_float();

    int channel = cur_scale.getType().cast<RankedTensorType>().getNumElements();
    std::vector<float> scale_v(channel);
    std::vector<float> bias_v(channel);
    for (int i = 0; i < channel; ++i) {
      scale_v[i] = cur_scale_f32->at(i) * next_scale_f32->at(i);
      bias_v[i] =
          cur_bias_f32->at(i) * next_scale_f32->at(i) + next_bias_f32->at(i);
    }

    auto new_scale = WeightOp::create_float(nextOp, "merged_scale", scale_v,
                                            {channel}, cur_storage_type);
    auto new_bias = WeightOp::create_float(nextOp, "merged_bias", bias_v,
                                           {channel}, cur_storage_type);
    nextOp->setOperand(1, new_scale);
    nextOp->setOperand(2, new_bias);

    rewriter.replaceOp(op, {op.getInput()});
    return success();
  }
};

struct ConstbinaryMergeToTopScale : public OpRewriterPatternEx<ScaleOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  ConstbinaryMergeToTopScale(MLIRContext *context, PatternBenefit benefit = 6)
      : OpRewriterPatternEx<ScaleOp>(context, "ConstbinaryMergeToTopScale",
                                     benefit) {}

  LogicalResult matchAndRewriteImpl(ScaleOp op,
                                    PatternRewriter &rewriter) const override {
    auto formerOp = op->getOperand(0).getDefiningOp();

    if (!formerOp->hasOneUse()) {
      return failure();
    }

    if (!isa<MulConstOp, AddConstOp>(formerOp)) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
      return failure();
    }

    auto scale = dyn_cast<WeightOp>(op.getScale().getDefiningOp());
    auto bias = dyn_cast<WeightOp>(op.getBias().getDefiningOp());
    auto scale_f32 = scale.read_as_float();
    auto bias_f32 = bias.read_as_float();

    int elem_num = scale.getType().cast<RankedTensorType>().getNumElements();
    std::vector<float> scale_v(elem_num);
    std::vector<float> bias_v(elem_num);

    if (isa<MulConstOp>(formerOp)) {
      auto mul_const_op = cast<MulConstOp>(formerOp);
      float value = mul_const_op.getConstVal().convertToDouble();
      for (int i = 0; i < elem_num; ++i) {
        scale_v[i] = scale_f32->at(i) * value;
      }
      auto new_scale = WeightOp::create_float(
          op, "constbinary_merged_to_scale", scale_v, {elem_num}, storage_type);
      op->setOperand(0, mul_const_op.getInput());
      op->setOperand(1, new_scale);
      op->setOperand(2, bias);
      rewriter.eraseOp(mul_const_op);
      return success();
    }

    if (isa<AddConstOp>(formerOp)) {
      auto add_const_op = cast<AddConstOp>(formerOp);
      float value = add_const_op.getConstVal().convertToDouble();
      for (int i = 0; i < elem_num; ++i) {
        bias_v[i] += scale_f32->at(i) * value;
      }
      auto bias_type = RankedTensorType::get({elem_num}, rewriter.getF32Type());
      auto new_bias =
          WeightOp::create(op, "constbinary_merged_to_bias", bias_v, bias_type);
      op->setOperand(0, add_const_op.getInput());
      op->setOperand(1, scale);
      op->setOperand(2, new_bias);
      rewriter.eraseOp(add_const_op);
      return success();
    }
    return failure();
  }
};

struct TopScaleMergeToBatchNorm : public OpRewriterPatternEx<ScaleOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  TopScaleMergeToBatchNorm(MLIRContext *context, PatternBenefit benefit = 9)
      : OpRewriterPatternEx<ScaleOp>(context, "TopScaleMergeToBatchNorm",
                                     benefit) {}

  LogicalResult matchAndRewriteImpl(ScaleOp op,
                                    PatternRewriter &rewriter) const override {
    auto formerOp = op.getInput().getDefiningOp();
    if (!formerOp->getResult(0).hasOneUse() || !isa<BatchNormOp>(formerOp)) {
      return failure();
    }
    auto bn_op = cast<BatchNormOp>(formerOp);
    if (bn_op.getDoRelu()) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
      return failure();
    }

    auto cur_scale_op = dyn_cast<WeightOp>(op.getScale().getDefiningOp());
    auto cur_bias_op = dyn_cast<WeightOp>(op.getBias().getDefiningOp());
    auto cur_scale_f32 = cur_scale_op.read_as_float();
    auto cur_bias_f32 = cur_bias_op.read_as_float();

    auto bn_mean_op = dyn_cast<WeightOp>(bn_op.getMean().getDefiningOp());
    auto bn_variance_op =
        dyn_cast<WeightOp>(bn_op.getVariance().getDefiningOp());
    auto bn_mean_f32 = bn_mean_op.read_as_float();
    auto bn_variance_f32 = bn_variance_op.read_as_float();

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

    auto bn_mean = WeightOp::create_float(bn_op, "merged_to_bn_mean", bn_mean_v,
                                          {channel}, storage_type);
    auto bn_variance = WeightOp::create_float(
        bn_op, "merged_to_bn_variance", bn_variance_v, {channel}, storage_type);
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
struct ScaleShapeAlign : public OpRewriterPatternEx<ScaleOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  ScaleShapeAlign(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewriterPatternEx<ScaleOp>(context, "ScaleShapeAlign", benefit) {}

  LogicalResult matchAndRewriteImpl(ScaleOp op,
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

struct TopScaleMergeToMatMul : public OpRewriterPatternEx<ScaleOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  TopScaleMergeToMatMul(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewriterPatternEx<ScaleOp>(context, "TopScaleMergeToMatMul",
                                     benefit) {}
  LogicalResult matchAndRewriteImpl(ScaleOp op,
                                    PatternRewriter &rewriter) const override {
    auto preOp = op.getInput().getDefiningOp();
    if (!preOp->hasOneUse() || !isa<MatMulOp>(preOp)) {
      return failure();
    }
    if (isa<MatMulOp>(preOp) && preOp->getAttr("do_relu").cast<BoolAttr>() ==
                                    rewriter.getBoolAttr(true)) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
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
    auto weight_data = weight.read_as_float();
    auto scale_data = scale.read_as_float();
    auto right_shape = module::getShape(matmulOp.getRight());
    auto N = scale.getType().cast<RankedTensorType>().getNumElements();
    assert(right_shape[1] == N);
    for (int k = 0; k < right_shape[0]; ++k) {
      for (int n = 0; n < right_shape[1]; ++n) {
        weight_data->at(k * right_shape[1] + n) *= scale_data->at(n);
      }
    }
    auto new_weight =
        WeightOp::create_float(matmulOp, "merged_scale_to_matmul", *weight_data,
                               {right_shape[0], right_shape[1]}, storage_type);
    matmulOp.setOperand(1, new_weight);

    // merge bias into matmul's bias
    auto bias_data = bias.read_as_float();
    std::vector<float> new_bias_v(N, 0);
    new_bias_v.assign(bias_data->begin(), bias_data->end());
    auto bias_type = RankedTensorType::get({N}, rewriter.getF32Type());
    if (!module::isNone(matmulOp.getBias())) {
      auto matmul_bias_data =
          dyn_cast<WeightOp>(matmulOp.getBias().getDefiningOp())
              .read_as_float();
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

struct FuseScaleIntoConv : public OpRewriterPatternEx<ScaleOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  FuseScaleIntoConv(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewriterPatternEx<ScaleOp>(context, "FuseScaleIntoConv", benefit) {}
  LogicalResult matchAndRewriteImpl(ScaleOp op,
                                    PatternRewriter &rewriter) const override {
    auto preOp = op.getInput().getDefiningOp();
    if (!preOp->hasOneUse() || !isa<ConvOp>(preOp)) {
      return failure();
    }
    auto convOp = cast<ConvOp>(preOp);
    if (convOp.getDoRelu()) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
      return failure();
    }
    auto c = module::getShape(convOp.getOutput())[1];
    auto scale = dyn_cast<WeightOp>(op.getScale().getDefiningOp());
    auto sBias = dyn_cast<WeightOp>(op.getBias().getDefiningOp());
    if (!sBias) {
      return failure();
    }
    std::vector<float_t> scaleVec(c, 1);
    if (scale) {
      auto scaleShape = module::getShape(scale);
      auto scaleData = scale.read_as_float();
      scaleVec.assign(scaleData->begin(), scaleData->end());
      if (std::find(scaleShape.begin(), scaleShape.end(), c) ==
              scaleShape.end() &&
          scaleVec.size() != c) {
        return failure();
      }
      auto filterOp = dyn_cast<WeightOp>(convOp.getFilter().getDefiningOp());
      if (!filterOp) { // filter may be not WeightOp
        return failure();
      }

      auto filterData = filterOp.read_as_float();
      auto filterShape = module::getShape(convOp.getFilter());
      std::vector<float_t> newFilter(filterData->size(), 0);
      uint32_t innerSize = filterData->size() / c;
      for (uint32_t i = 0; i < c; ++i) {
        for (uint32_t j = 0; j < innerSize; ++j) {
          newFilter.at(i * innerSize + j) =
              filterData->at(i * innerSize + j) * scaleVec.at(i);
        }
      }
      auto new_filter = WeightOp::create_float(filterOp, "", newFilter,
                                               filterShape, storage_type);
      convOp.setOperand(1, new_filter);
    }
    if (sBias) {
      // merge SBias into conv's bias
      auto sBiasShape = module::getShape(sBias);
      auto sBiasData = sBias.read_as_float();
      if (std::find(sBiasShape.begin(), sBiasShape.end(), c) ==
              sBiasShape.end() &&
          sBiasData->size() != c) {
        return failure();
      }
      std::vector<float_t> newBiasVec(c, 0);
      newBiasVec.assign(sBiasData->begin(), sBiasData->end());
      auto newBiasType = RankedTensorType::get({c}, rewriter.getF32Type());
      if (!module::isNone(convOp.getBias())) {
        auto cBiasOp = dyn_cast<WeightOp>(convOp.getBias().getDefiningOp());
        if (!cBiasOp) { // filter may be not WeightOp
          return failure();
        }
        auto cBiasData = cBiasOp.read_as_float();
        for (int i = 0; i < c; ++i) {
          newBiasVec[i] += cBiasData->at(i) * scaleVec[i];
        }
        cBiasOp.update(newBiasVec, c);
      } else {
        auto newBiasOp = WeightOp::create(
            convOp, module::getName(sBias, 0).str(), newBiasVec, newBiasType);
        convOp.setOperand(2, newBiasOp);
      }
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
  results.insert<TopMultiScaleMergeToOne, TopScaleMergeToBatchNorm,
                 ScaleShapeAlign, ConstbinaryMergeToTopScale,
                 TopScaleMergeToMatMul>(context);
}

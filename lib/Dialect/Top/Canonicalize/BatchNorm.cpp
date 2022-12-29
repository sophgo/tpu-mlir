//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"


using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;


struct TopBatchNormToScale : public OpRewritePattern<BatchNormOp> {
  using OpRewritePattern::OpRewritePattern;
  TopBatchNormToScale(MLIRContext *context, PatternBenefit benefit = 9)
      : OpRewritePattern<BatchNormOp>(context, benefit) {}

  LogicalResult matchAndRewrite(BatchNormOp op,
                                PatternRewriter &rewriter) const override {

    auto mean = cast<WeightOp>(op.mean().getDefiningOp());
    auto variance = cast<WeightOp>(op.variance().getDefiningOp());
    auto mean_f32 = mean.read<float>();
    auto variance_f32 = variance.read<float>();

    auto shape = module::getShape(op.input());
    auto channel = shape.size() > 1 ? shape[1] : shape[0];

    std::shared_ptr<std::vector<float>> gamma_f32;
    if (auto gamma = dyn_cast<WeightOp>(op.gamma().getDefiningOp())) {
      gamma_f32 = gamma.read<float>();
    } else {
      gamma_f32 = std::make_shared<std::vector<float>>(channel, 1.0f);
    }
    std::shared_ptr<std::vector<float>> beta_f32;
    if (auto beta = dyn_cast<WeightOp>(op.beta().getDefiningOp())) {
      beta_f32 = beta.read<float>();
    } else {
      beta_f32 = std::make_shared<std::vector<float>>(channel, 0.0f);
    }

    std::vector<float> scale(channel);
    std::vector<float> bias(channel);

    // constructe scale and bias by params of BatchNorm
    auto eps = op.epsilon().convertToDouble();
    for (int i = 0; i < channel; ++i) {
      scale[i] = 1 / std::sqrt(variance_f32->at(i) + eps) * gamma_f32->at(i);
      bias[i] = -mean_f32->at(i) * scale[i] + beta_f32->at(i);
    }

    auto scale_type = RankedTensorType::get({channel}, rewriter.getF32Type());
    auto scale_op = WeightOp::create(op, "scale", scale, scale_type);
    auto bias_type = RankedTensorType::get({channel}, rewriter.getF32Type());
    auto bias_op = WeightOp::create(op, "bias", bias, bias_type);
    // replace the BatchNorm Op
    rewriter.replaceOpWithNewOp<ScaleOp>(
        op, op.output().getType(), ValueRange{op.input(), scale_op, bias_op});
    return success();
  }
};

void BatchNormOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<TopBatchNormToScale>(context);
}

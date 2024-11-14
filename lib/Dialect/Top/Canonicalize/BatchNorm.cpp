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

struct TopBatchNormToScale : public OpRewriterPatternEx<BatchNormOp> {
  TopBatchNormToScale(MLIRContext *context, PatternBenefit benefit = 9)
      : OpRewriterPatternEx<BatchNormOp>(context, "TopBatchNormToScale",
                                         benefit) {}

protected:
  LogicalResult matchAndRewriteImpl(BatchNormOp op,
                                    PatternRewriter &rewriter) const override {
    auto mean = cast<WeightOp>(op.getMean().getDefiningOp());
    auto variance = cast<WeightOp>(op.getVariance().getDefiningOp());
    auto mean_f32 = mean.read_as_float();
    auto variance_f32 = variance.read_as_float();

    auto shape = module::getShape(op.getInput());
    auto channel = shape.size() > 1 ? shape[1] : shape[0];

    std::shared_ptr<std::vector<float>> gamma_f32;
    if (auto gamma = dyn_cast<WeightOp>(op.getGamma().getDefiningOp())) {
      gamma_f32 = gamma.read_as_float();
    } else {
      gamma_f32 = std::make_shared<std::vector<float>>(channel, 1.0f);
    }
    std::shared_ptr<std::vector<float>> beta_f32;
    if (auto beta = dyn_cast<WeightOp>(op.getBeta().getDefiningOp())) {
      beta_f32 = beta.read_as_float();
    } else {
      beta_f32 = std::make_shared<std::vector<float>>(channel, 0.0f);
    }

    std::vector<float> scale(channel);
    std::vector<float> bias(channel);

    // construct scale and bias by params of BatchNorm
    auto eps = op.getEpsilon().convertToDouble();
    for (int i = 0; i < channel; ++i) {
      scale[i] = 1 / std::sqrt(variance_f32->at(i) + eps) * gamma_f32->at(i);
      bias[i] = -mean_f32->at(i) * scale[i] + beta_f32->at(i);
    }
    auto storage_type = module::getStorageType(op.getOutput());

    auto scale_op =
        WeightOp::create_float(op, "scale", scale, {channel}, storage_type);
    auto bias_op =
        WeightOp::create_float(op, "bias", bias, {channel}, storage_type);

    // replace the BatchNorm Op
    rewriter.replaceOpWithNewOp<ScaleOp>(
        op, op.getOutput().getType(),
        ValueRange{op.getInput(), scale_op, bias_op});
    return success();
  }
};

void BatchNormOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<TopBatchNormToScale>(context);
}

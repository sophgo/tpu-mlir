//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-prelu"
namespace tpu_mlir {
namespace cv18xx {
void PReluLowering::LoweringINT8(PatternRewriter &rewriter, top::PReluOp op,
                                 bool asymmetric) const {
  int64_t N, C, H, W;
  module::getNCHW(op.getOutput(), N, C, H, W);
  auto slope_shape = module::getShape(op.getSlope());
  auto num_slope = module::getNumElements(op.getSlope());
  assert(num_slope == C);

  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;

  // quantize positive
  auto threshold_x = module::getThreshold(op.getInput());
  auto threshold_y = module::getThreshold(op.getOutput());
  double qscale_pos = threshold_x / threshold_y;
  int64_t multiplier_pos, rshift_pos;
  if (std::fabs(threshold_x - threshold_y) <
      1e-5 * std::min(threshold_x, threshold_y)) {
    // no positive scale
    rshift_pos = 0;
    multiplier_pos = 1;
  } else {
    getRShiftAndMultiplierFromQScale(qscale_pos, &multiplier_pos, &rshift_pos,
                                     false);
  }

  // quantize negtive
  auto slopeOp = cast<top::WeightOp>(op.getSlope().getDefiningOp());
  auto slope_f32 = slopeOp.read<float>();
  std::vector<int8_t> slope_int8(num_slope);
  auto scale_max = std::fabs(slope_f32->at(0) * qscale_pos);
  for (int idx = 1; idx < num_slope; idx++) {
    auto scale = std::fabs(slope_f32->at(idx) * qscale_pos);
    if (scale > scale_max) {
      scale_max = scale;
    }
  }
  int64_t scalei, rshifti;
  getRShiftAndMultiplierFromQScale(scale_max, &scalei, &rshifti);

  for (int idx = 0; idx < num_slope; idx++) {
    slope_int8[idx] = quantizeFilterRShift(slope_f32->at(idx), threshold_y,
                                           threshold_x, rshifti);
  }

  operands.push_back(op.getInput());
  auto new_type = RankedTensorType::get(slope_shape, rewriter.getI8Type());
  auto new_slope = top::WeightOp::create(op, "slope_i8", slope_int8, new_type);
  operands.push_back(new_slope);

  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(rshifti)));
  attrs.push_back(rewriter.getNamedAttr(
      "rshift_pos", rewriter.getSI32IntegerAttr(rshift_pos)));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier_pos", rewriter.getSI32IntegerAttr(multiplier_pos)));
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::PReluOp>(op, newType, operands, attrs);
}

void PReluLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::PReluOp op) const {
  lowering_common_bf16<tpu::PReluOp>(rewriter, op);
}
} // namespace cv18xx
} // namespace tpu_mlir

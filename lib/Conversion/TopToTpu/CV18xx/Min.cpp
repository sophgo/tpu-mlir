//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-min"
namespace tpu_mlir {
namespace cv18xx {
void MinLowering::LoweringINT8(PatternRewriter &rewriter, top::MinOp op,
                               bool asymmetric) const {
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  int64_t o_zp;
  double o_scale;
  bool sign = true;
  module::getScaleAndZeroPoint(op.getOutput(), o_scale, o_zp, sign, false);
  std::vector<int> coeff_v(nInputs, 1);
  std::vector<float> qscale(nInputs);
  for (int i = 0; i < nInputs; i++) {
    double i_scale;
    int64_t i_zp;
    auto input = op->getOperand(i);
    operands.push_back(input);
    module::getScaleAndZeroPoint(input, i_scale, i_zp, sign, false);
    auto scale_f = i_scale / o_scale;
    qscale[i] = coeff_v[i] * scale_f;
  }

  float max_qscale = 0.0;
  for (auto &q : qscale) {
    if (max_qscale < std::abs(q)) {
      max_qscale = std::abs(q);
    }
  }
  std::vector<int64_t> rshift_v(1);
  std::vector<int64_t> multiplier_v(nInputs, 1);
  int64_t multiplier = 0;
  int64_t shift = 0;
  getRShiftAndMultiplierFromQScale(max_qscale, &multiplier, &shift, false);

  rshift_v[0] = shift;
  for (int i = 0; i < nInputs; ++i) {
    multiplier_v[i] = getMultiplierI8FromQScaleAndRShift(qscale[i], shift);
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "multipliers", rewriter.getI64ArrayAttr(multiplier_v)));
  attrs.push_back(
      rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(rshift_v)));
  auto newType = getQuantInt8Type(op.getOutput(), false);
  rewriter.replaceOpWithNewOp<tpu::MinOp>(op.getOperation(), newType, operands,
                                          attrs);
  return;
}

void MinLowering::LoweringBF16(PatternRewriter &rewriter, top::MinOp op) const {
  lowering_common_bf16<tpu::MinOp>(rewriter, op);
}
} // namespace cv18xx
} // namespace tpu_mlir

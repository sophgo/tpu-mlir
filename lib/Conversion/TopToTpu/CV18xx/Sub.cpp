//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-sub"

namespace tpu_mlir {
namespace cv18xx {

void SubLowering::LoweringINT8(PatternRewriter &rewriter, top::SubOp op,
                               bool asymmetric) const {
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto v : op->getOperands()) {
    if (module::isWeight(v)) {
      LoweringBF16(rewriter, op);
      return;
    }
  }
  std::vector<int64_t> rshift_v(1);
  std::vector<int64_t> multiplier_v(nInputs, 1);
  std::vector<float> qscale(nInputs, 1.0);
  float max_qscale = 0.0;
  assert(nInputs == 2);
  auto coeff_v = module::getF64Array(op.getCoeff(), 2, 1.0);
  assert(coeff_v->at(0) == 1 && coeff_v->at(1) == 1);
  double o_scale = module::getThreshold(op.getOutput());
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    operands.push_back(input);
    double i_scale = module::getThreshold(input);
    auto scale_f = i_scale / o_scale;
    qscale[i] = coeff_v->at(i) * scale_f;
  }

  for (auto &q : qscale) {
    if (max_qscale < std::abs(q)) {
      max_qscale = std::abs(q);
    }
  }
  int64_t multiplier = 0;
  int64_t shift = 0;
  getRShiftAndMultiplierFromQScale(max_qscale, &multiplier, &shift, false);

  rshift_v[0] = shift;
  for (int i = 0; i < nInputs; ++i) {
    multiplier_v[i] = getMultiplierI8FromQScaleAndRShift(qscale[i], shift);
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
  attrs.push_back(rewriter.getNamedAttr("relu_limit", op.getReluLimitAttr()));
  attrs.push_back(
      rewriter.getNamedAttr("coeff", rewriter.getF64ArrayAttr({1, 1})));
  attrs.push_back(rewriter.getNamedAttr(
      "multipliers", rewriter.getI64ArrayAttr(multiplier_v)));
  attrs.push_back(
      rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(rshift_v)));
  auto newType = getQuantInt8Type(op.getOutput());
  // todo  if prod(shape0) < prod(shape1) result mul -1 here
  attrs.push_back(rewriter.getNamedAttr("is_reverse", op.getIsReverseAttr()));
  rewriter.replaceOpWithNewOp<tpu::SubOp>(op.getOperation(), newType, operands,
                                          attrs);
  return;
}

void SubLowering::LoweringBF16(PatternRewriter &rewriter, top::SubOp op) const {
  lowering_common_bf16<tpu::SubOp>(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir

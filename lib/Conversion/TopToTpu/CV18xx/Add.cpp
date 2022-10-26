//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-add"

namespace tpu_mlir {
namespace cv18xx {
void AddLowering::LoweringINT8(PatternRewriter &rewriter, top::AddOp op,
                                bool asymmetric) const {
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  int64_t o_zp;
  double o_scale;
  bool sign = true;
  Quant::getScaleAndZeroPoint(op.output(), o_scale, o_zp, sign, false);
  auto coeff_v = Module::getF64Array(op.coeff(), nInputs, 1.0);

  std::vector<float> qscale(nInputs);
  for (int i = 0; i < nInputs; i++) {
    double i_scale;
    int64_t i_zp;
    auto input = op->getOperand(i);
    operands.push_back(input);
    Quant::getScaleAndZeroPoint(input, i_scale, i_zp, sign, false);
    auto scale_f = i_scale / o_scale;
    qscale[i] = coeff_v->at(i) * scale_f;
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
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.do_reluAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "multipliers", rewriter.getI64ArrayAttr(multiplier_v)));
  attrs.push_back(
      rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(rshift_v)));
  auto newType = Quant::getQuantInt8Type(op.output(), false);
  rewriter.replaceOpWithNewOp<tpu::AddOp>(op.getOperation(), newType, operands, attrs);
  return;

}
void AddLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::AddOp op) const {
  llvm_unreachable("Not supported now");
}

} // namespace cv18xx
} // namespace tpu_mlir

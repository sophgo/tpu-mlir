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

#define DEBUG_TYPE "lowering-mul"

namespace tpu_mlir {
namespace cv18xx {
void MulLowering::LoweringINT8(PatternRewriter &rewriter, top::MulOp op,
                               bool asymmetric) const {
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  int64_t o_zp;
  double o_scale;
  bool sign = true;
  Quant::getScaleAndZeroPoint(op.output(), o_scale, o_zp, sign, false);
  double scalef = 1.0;
  for (int i = 0; i < nInputs; i++) {
    double i_scale;
    int64_t i_zp;
    auto input = op->getOperand(i);
    operands.push_back(input);
    Quant::getScaleAndZeroPoint(input, i_scale, i_zp, sign, false);
    scalef *= i_scale;
  }
  scalef /= o_scale;

  //For quant compare npz result use.
  int multiplier;
  int rshift;
  get_scale_and_shift(scalef, multiplier, rshift, 8);

  //This quantinization is for codegen/backend use,because cv18xx mulOp should quant in "qdm" mode.

  int64_t multiplier_cg;
  int64_t rshift_cg;
  getRShiftAndMultiplierFromQScale(scalef, &multiplier_cg, &rshift_cg, true);

  /*Note:Due to the different way to get multiplier and rshift,the final result compare between
   "_int8_sym_tpu_out.npz" and "_int8_sym_model_out.npz" maybe a little different.*/
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.do_reluAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(rshift)));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier_cg", rewriter.getI64IntegerAttr(multiplier_cg)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift_cg", rewriter.getI64IntegerAttr(rshift_cg)));
  auto newType = Quant::getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::MulOp>(op, newType, operands, attrs);
}

void MulLowering::LoweringBF16(PatternRewriter &rewriter, top::MulOp op) const {
  lowering_common_bf16<tpu::MulOp>(rewriter, op);
}
}
}

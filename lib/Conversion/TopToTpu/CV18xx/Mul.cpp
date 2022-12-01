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
  double scale_i;
  int64_t i_zp;
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    if (auto constOp = dyn_cast<top::WeightOp>(input.getDefiningOp())) {
      auto constF32 = constOp.read<float>();
      // float fmax, fmin;
      // findMinMax(constF32->data(), constF32->size(), &fmin, &fmax);
      // bool cSign = (fmin < 0);
      //float fqmax = cSign ? 127 : 255;
      float fmax = findMaxabs(constF32->data(), constF32->size());
      bool cSign = true;
      float fqmax = 127.0;
      auto filter_type = input.getType().cast<RankedTensorType>();
      auto new_type = RankedTensorType::get(filter_type.getShape(),
                                            rewriter.getIntegerType(8, cSign));
      scale_i = fmax / fqmax;
      if (cSign) {
        auto constI8 = std::make_shared<std::vector<int8_t>>(constF32->size());
        std::transform(
            constF32->begin(), constF32->end(), constI8->begin(),
            [&](const float cf32) { return Quant::to_int8(cf32 / scale_i); });
        auto new_filter =
            top::WeightOp::create(constOp, "i8", *constI8, new_type);
        operands.push_back(new_filter);
      } else {
        auto constU8 = std::make_shared<std::vector<uint8_t>>(constF32->size());
        std::transform(
            constF32->begin(), constF32->end(), constU8->begin(),
            [&](const float cf32) { return Quant::to_uint8(cf32 / scale_i); });
        auto new_filter =
            top::WeightOp::create(constOp, "u8", *constU8, new_type);
        operands.push_back(new_filter);
      }
    } else {
      Quant::getScaleAndZeroPoint(input, scale_i, i_zp, sign, false);
      operands.push_back(input);
    }
    scalef *= scale_i;
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
  for (int i = 0, n = op.getNumOperands(); i < n; ++i) {
    if (auto constOp =
            dyn_cast<top::WeightOp>(op.getOperand(i).getDefiningOp())) {
      op.setOperand(i, constOp.clone_bf16(op));
    }
  }
  lowering_common_bf16<tpu::MulOp>(rewriter, op);
}
}
}

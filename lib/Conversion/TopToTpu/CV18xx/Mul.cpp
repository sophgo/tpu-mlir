//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-mul"

namespace tpu_mlir {
namespace cv18xx {
void MulLowering::LoweringINT8(PatternRewriter &rewriter, top::MulOp op,
                               bool asymmetric) const {
  // for convert from DivOp
  auto div_v = op.getInputs()[1];
  if (!module::isCalibratedType(div_v) && !module::isUniformQuantized(div_v)) {
    LoweringBF16(rewriter, op);
    return;
  }

  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  int64_t o_zp;
  double o_scale;
  bool sign = true;
  module::getScaleAndZeroPoint(op.getOutput(), o_scale, o_zp, sign, false);
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
      // float fqmax = cSign ? 127 : 255;
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
            [&](const float cf32) { return to_int8(cf32 / scale_i); });
        auto new_filter =
            top::WeightOp::create(constOp, "i8", *constI8, new_type);
        operands.push_back(new_filter);
      } else {
        auto constU8 = std::make_shared<std::vector<uint8_t>>(constF32->size());
        std::transform(
            constF32->begin(), constF32->end(), constU8->begin(),
            [&](const float cf32) { return to_uint8(cf32 / scale_i); });
        auto new_filter =
            top::WeightOp::create(constOp, "u8", *constU8, new_type);
        operands.push_back(new_filter);
      }
    } else {
      module::getScaleAndZeroPoint(input, scale_i, i_zp, sign, false);
      operands.push_back(input);
    }
    scalef *= scale_i;
  }
  scalef /= o_scale;

  int64_t multiplier;
  int64_t rshift;
  getRShiftAndMultiplierFromQScale(scalef, &multiplier, &rshift, true);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(rshift)));
  attrs.push_back(rewriter.getNamedAttr(
      "quant_mode",
      tpu::RequantModeAttr::get(getContext(), tpu::RequantMode::QDM)));
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::MulOp>(op, newType, operands, attrs);
}

void MulLowering::LoweringBF16(PatternRewriter &rewriter, top::MulOp op) const {
  lowering_common_bf16<tpu::MulOp>(rewriter, op);
}
} // namespace cv18xx
} // namespace tpu_mlir

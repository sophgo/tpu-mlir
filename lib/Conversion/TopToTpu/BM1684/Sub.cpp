//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void SubLowering::LoweringINT8(PatternRewriter &rewriter, top::SubOp op,
                               bool asymmetric) const {
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  std::vector<int64_t> rshift_v(nInputs);
  std::vector<int64_t> multiplier_v(nInputs, 1);
  std::vector<double> coeff_v(nInputs, 1.0);
  int64_t o_zp;
  double o_scale;
  module::getScaleAndZeroPoint(op.getOutput(), o_scale, o_zp, asymmetric);

  if (op.getCoeff().has_value()) {
    int idx = 0;
    auto coeff = op.getCoeff().value();
    for (auto v : coeff) {
      coeff_v[idx++] = v.cast<FloatAttr>().getValueAsDouble();
    }
  }
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    int64_t in_zp;
    double in_scale;
    if (auto ConstOp = dyn_cast<top::WeightOp>(input.getDefiningOp())) {
      auto constF32 = ConstOp.read<float>();
      float fmax, fmin;
      findMinMax(constF32->data(), constF32->size(), &fmin, &fmax);
      fmax = std::max(fabs(fmax), fabs(fmin));
      bool ConstSign = (fmin < 0);
      float fqmax = ConstSign ? 127 : 255;
      auto filter_type = input.getType().cast<RankedTensorType>();
      auto new_type = RankedTensorType::get(
          filter_type.getShape(), rewriter.getIntegerType(8, ConstSign));
      in_scale = fmax / fqmax;
      if (ConstSign) {
        auto constI8 = std::make_shared<std::vector<int8_t>>(constF32->size());
        std::transform(
            constF32->begin(), constF32->end(), constI8->begin(),
            [&](const float cf32) { return to_int8(cf32 / in_scale); });
        auto new_filter =
            top::WeightOp::create(ConstOp, "i8", *constI8, new_type);
        operands.push_back(new_filter);
      } else {
        auto constU8 = std::make_shared<std::vector<uint8_t>>(constF32->size());
        std::transform(
            constF32->begin(), constF32->end(), constU8->begin(),
            [&](const float cf32) { return to_uint8(cf32 / in_scale); });
        auto new_filter =
            top::WeightOp::create(ConstOp, "U8", *constU8, new_type);
        operands.push_back(new_filter);
      }
    } else {
      operands.push_back(input);
      module::getScaleAndZeroPoint(input, in_scale, in_zp, asymmetric);
    }
    rshift_v[i] =
        calRightShiftNumUseCblas(coeff_v[i], in_scale, o_scale, BITS_INT8);
    rshift_v[i] = rshift_v[i] < 0 ? 0 : rshift_v[i];
    float scale = 1.0 * (1 << rshift_v[i]) * in_scale / o_scale;
    int8_t multiplier_int8 = 0;
    float coeff = coeff_v[i];
    quantizeToInt8(&coeff, &multiplier_int8, 1, scale);
    if (multiplier_int8 < 0 || multiplier_int8 > 127) {
      lowering_common_f32<tpu::SubOp>(rewriter, op);
      return;
    }
    multiplier_v[i] = (double)multiplier_int8;
  }
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "multipliers", rewriter.getI64ArrayAttr(multiplier_v)));
  attrs.push_back(
      rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(rshift_v)));
  auto newType = getQuantInt8Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::SubOp>(op, newType, operands, attrs);
}

void SubLowering::LoweringF32(PatternRewriter &rewriter, top::SubOp op) const {
  lowering_common_f32<tpu::SubOp>(rewriter, op);
}

} // namespace bm1684
} // namespace tpu_mlir

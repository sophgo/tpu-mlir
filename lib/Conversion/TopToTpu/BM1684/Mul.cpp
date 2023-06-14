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

void MulLowering::LoweringF32(PatternRewriter &rewriter, top::MulOp op) const {
  lowering_common_f32<tpu::MulOp>(rewriter, op);
}

void MulLowering::LoweringINT8(PatternRewriter &rewriter, top::MulOp op,
                               bool asymmetric) const {
  const int nInputs = op->getNumOperands();
  std::vector<Value> operands;
  double scale = 1;
  int64_t zp_o = 0;
  double scale_o = 1;
  module::getScaleAndZeroPoint(op.getOutput(), scale_o, zp_o, asymmetric);

  double scale_i;
  int64_t zp;
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    if (auto constOp = dyn_cast<top::WeightOp>(input.getDefiningOp())) {
      auto constF32 = constOp.read<float>();
      float fmax, fmin;
      findMinMax(constF32->data(), constF32->size(), &fmin, &fmax);
      fmax = std::max(fabs(fmax), fabs(fmin));
      bool cSign = (fmin < 0);
      float fqmax = cSign ? 127 : 255;
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
      module::getScaleAndZeroPoint(input, scale_i, zp, asymmetric);
      operands.push_back(input);
    }
    scale *= scale_i;
  }

  scale /= scale_o;

  int multiplier;
  int rshift;
  get_scale_and_shift(scale, multiplier, rshift, 8);
  rshift = std::min(std::max(0, rshift), 31);
  if (multiplier < 0 || multiplier > 127) {
    lowering_common_f32<tpu::MulOp>(rewriter, op);
    return;
  }
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(rshift)));
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::MulOp>(op, newType, operands, attrs);
}

} // namespace bm1684
} // namespace tpu_mlir

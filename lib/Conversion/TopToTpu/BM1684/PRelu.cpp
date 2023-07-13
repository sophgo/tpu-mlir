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

void PReluLowering::LoweringF32(PatternRewriter &rewriter,
                                top::PReluOp op) const {
  lowering_common_f32<tpu::PReluOp>(rewriter, op);
}

void PReluLowering::LoweringINT8(PatternRewriter &rewriter, top::PReluOp op,
                                 bool asymmetric) const {
  if (asymmetric == false) {
    int64_t N, C, H, W;
    module::getNCHW(op.getOutput(), N, C, H, W);
    auto slope_shape = module::getShape(op.getSlope());
    auto num_slope = module::getNumElements(op.getSlope());
    assert(num_slope == C);
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    auto slopeOp = cast<top::WeightOp>(op.getSlope().getDefiningOp());
    auto slope_f32 = slopeOp.read<float>();
    auto scale_max = findMaxabs(slope_f32->data(), slope_f32->size());
    std::vector<int8_t> slope_int8(num_slope, 0);
    int64_t scalei, rshifti;
    getRShiftAndMultiplierFromQScale(scale_max, &scalei, &rshifti);
    for (int idx = 0; idx < num_slope; idx++) {
      slope_int8[idx] =
          getMultiplierI8FromQScaleAndRShift(slope_f32->at(idx), rshifti);
    }
    operands.push_back(op.getInput());
    auto new_type = RankedTensorType::get(slope_shape, rewriter.getI8Type());
    auto new_slope =
        top::WeightOp::create(op, "slope_i8", slope_int8, new_type);
    operands.push_back(new_slope);
    for (auto &attr : op->getAttrs()) {
      attrs.push_back(attr);
    }
    attrs.push_back(
        rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(rshifti)));
    auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
    rewriter.replaceOpWithNewOp<tpu::PReluOp>(op, newType, operands, attrs);
  } else {
    LoweringF32(rewriter, op);
  }
}

} // namespace bm1684
} // namespace tpu_mlir

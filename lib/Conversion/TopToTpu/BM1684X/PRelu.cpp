//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {
void PReluLowering::LoweringINT4(PatternRewriter &rewriter, top::PReluOp op,
                                 bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
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
    std::vector<int8_t> slope_int8(num_slope);
    auto scale_max = std::abs(slope_f32->at(0));
    for (int idx = 1; idx < num_slope; idx++) {
      auto scale = std::abs(slope_f32->at(idx));
      if (scale > scale_max) {
        scale_max = scale;
      }
    }
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
    LoweringF16(rewriter, op);
  }
}

void PReluLowering::LoweringF32(PatternRewriter &rewriter,
                                top::PReluOp op) const {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  auto slope_num = module::getNumElements(op.getSlope());
  bool channel_share = false;
  if (slope_num == 1) {
    channel_share = true;
  }

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(rewriter.getNamedAttr("channel_shared",
                                        rewriter.getBoolAttr(channel_share)));
  if (channel_share) {
    auto slopeOp = cast<top::WeightOp>(op.getSlope().getDefiningOp());
    auto slope_f32 = slopeOp.read<float>();
    float *slope_data = slope_f32->data();
    float slope_val = (float)(*slope_data);
    attrs.push_back(rewriter.getNamedAttr("slope_val",
                                          rewriter.getF64FloatAttr(slope_val)));
  }
  Value newValue;
  auto newOp = rewriter.create<tpu::PReluOp>(
      op->getLoc(), op.getOutput().getType(), operands, attrs);
  newValue = newOp.getOutput();
  rewriter.replaceOp(op, {newValue});
}

void PReluLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::PReluOp op) const {
  lowering_common_bf16<tpu::PReluOp>(rewriter, op);
}

void PReluLowering::LoweringF16(PatternRewriter &rewriter,
                                top::PReluOp op) const {
  lowering_common_f16<tpu::PReluOp>(rewriter, op);
}

void PReluLowering::LoweringF8(PatternRewriter &rewriter,
                               top::PReluOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void PReluLowering::LoweringQuantized(PatternRewriter &rewriter,
                                      top::PReluOp op) const {
  // UNREACHABLE_OP("Not Implemented", op);
  LoweringINT8(rewriter, op, false);
}

} // namespace bm1684x
} // namespace tpu_mlir

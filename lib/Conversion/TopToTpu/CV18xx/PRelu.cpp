//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-prelu"
namespace tpu_mlir {
namespace cv18xx {
void PReluLowering::LoweringINT8(PatternRewriter &rewriter, top::PReluOp op,
                                 bool asymmetric) const {
  int64_t N, C, H, W;
  Module::getNCHW(op.output(), N, C, H, W);
  auto src_shape = Module::getShape(op.input());
  auto slope_shape = Module::getShape(op.slope());
  auto num_slope = Module::getNumElements(op.slope());
  assert(num_slope == C);

  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;

  auto slopeOp = cast<top::WeightOp>(op.slope().getDefiningOp());
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

  operands.push_back(op.input());
  auto new_type = RankedTensorType::get(slope_shape, rewriter.getI8Type());
  auto new_slope = top::WeightOp::create(op, "slope_i8", slope_int8, new_type);
  operands.push_back(new_slope);

  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(rshifti)));
  auto newType = getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::PReluOp>(op, newType, operands, attrs);
}

void PReluLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::PReluOp op) const {
  lowering_common_bf16<tpu::PReluOp>(rewriter, op);
}
} // namespace cv18xx
} // namespace tpu_mlir

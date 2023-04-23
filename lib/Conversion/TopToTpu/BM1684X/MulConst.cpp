//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"
#define FP16_MAX 65504.0
#define FP16_MIN -65504.0
#define BF16_MAX 3.3895314e38
#define BF16_MIN -3.3895314e38

namespace tpu_mlir {
namespace bm1684x {

void MulConstLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::MulConstOp op) const {
  lowering_common_f32<tpu::MulConstOp>(rewriter, op);
}
void MulConstLowering::LoweringINT4(PatternRewriter &rewriter, top::MulConstOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void MulConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::MulConstOp op, bool asymmetric) const {
  double scale_i, scale_o;
  int64_t zp_i, zp_o;
  module::getScaleAndZeroPoint(op.getInput(), scale_i, zp_i, asymmetric);
  module::getScaleAndZeroPoint(op.getOutput(), scale_o, zp_o, asymmetric);
  auto scale = scale_i / scale_o * op.getConstVal().convertToDouble();
  int multiplier, rshift;
  get_scale_and_shift(scale, multiplier, rshift, 8);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(rshift)));
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::MulShiftOp>(op, newType,
                                               ValueRange{op.getInput()}, attrs);
}

void MulConstLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::MulConstOp op) const {
  auto const_v = op.getConstVal().convertToDouble();
  if (const_v > BF16_MAX || const_v < BF16_MIN)
    LoweringF32(rewriter, op);
  else
    lowering_common_bf16<tpu::MulConstOp>(rewriter, op);
}

void MulConstLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::MulConstOp op) const {
  auto const_v = op.getConstVal().convertToDouble();
  if (const_v > FP16_MAX || const_v < FP16_MIN)
    LoweringF32(rewriter, op);
  else
    lowering_common_f16<tpu::MulConstOp>(rewriter, op);
}

void MulConstLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::MulConstOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir

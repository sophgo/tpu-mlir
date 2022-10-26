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

void MulConstLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::MulConstOp op) const {
  lowering_common_f32<tpu::MulConstOp>(rewriter, op);
}

void MulConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::MulConstOp op, bool asymmetric) const {
  double scale_i, scale_o;
  int64_t zp_i, zp_o;
  Quant::getScaleAndZeroPoint(op.input(), scale_i, zp_i, asymmetric);
  Quant::getScaleAndZeroPoint(op.output(), scale_o, zp_o, asymmetric);
  auto scale = scale_i / scale_o * op.const_val().convertToDouble();
  int multiplier, rshift;
  get_scale_and_shift(scale, multiplier, rshift, 8);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(rshift)));
  auto newType = Quant::getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::MulShiftOp>(op, newType,
                                               ValueRange{op.input()}, attrs);
}

void MulConstLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::MulConstOp op) const {
  lowering_common_bf16<tpu::MulConstOp>(rewriter, op);
}

void MulConstLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::MulConstOp op) const {
  lowering_common_f16<tpu::MulConstOp>(rewriter, op);
}

void MulConstLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::MulConstOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir

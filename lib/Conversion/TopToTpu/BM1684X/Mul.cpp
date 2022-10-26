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

void MulLowering::LoweringF32(PatternRewriter &rewriter, top::MulOp op) const {
  lowering_common_f32<tpu::MulOp>(rewriter, op);
}

void MulLowering::LoweringINT8(PatternRewriter &rewriter, top::MulOp op,
                               bool asymmetric) const {
  if (asymmetric) {
    LoweringF32(rewriter, op);
    return;
  }
  const int nInputs = op->getNumOperands();
  std::vector<Value> operands;
  double scale;
  int64_t zp_o;
  double scale_o;
  Quant::getScaleAndZeroPoint(op.output(), scale_o, zp_o, asymmetric);

  double scale_i;
  int64_t zp;
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    operands.push_back(input);
    Quant::getScaleAndZeroPoint(input, scale_i, zp, asymmetric);
    if (i == 0)
      scale = scale_i;
    else
      scale *= scale_i;
  }

  scale /= scale_o;

  int multiplier;
  int rshift;
  get_scale_and_shift(scale, multiplier, rshift, 8);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.do_reluAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(rshift)));
  auto newType = Quant::getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::MulOp>(op, newType, operands, attrs);
}

void MulLowering::LoweringBF16(PatternRewriter &rewriter, top::MulOp op) const {
  lowering_common_bf16<tpu::MulOp>(rewriter, op);
}

void MulLowering::LoweringF16(PatternRewriter &rewriter, top::MulOp op) const {
  lowering_common_f16<tpu::MulOp>(rewriter, op);
}

void MulLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::MulOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir

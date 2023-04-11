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

void SubConstLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::SubConstOp op) const {
  lowering_common_f32<tpu::SubConstOp>(rewriter, op);
}

void SubConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::SubConstOp op, bool asymmetric) const {
  if (asymmetric)
    return LoweringF16(rewriter, op);

  auto in = op.getInput();
  auto out = op.getOutput();
  double in_scale, out_scale;

  int64_t in_zp, out_zp;
  module::getScaleAndZeroPoint(in, in_scale, in_zp, asymmetric);
  module::getScaleAndZeroPoint(out, out_scale, out_zp, asymmetric);
  int multiplier, rshift;
  get_scale_and_shift_positive(in_scale / out_scale, multiplier, rshift, 8);

  double const_b = op.getConstVal().convertToDouble();
  const_b = static_cast<int>(round(const_b / out_scale)) << rshift;

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    if (attr.getName() == "const_val") {
      attrs.push_back(rewriter.getNamedAttr("const_val",
                                            rewriter.getF64FloatAttr(const_b)));
    } else {
      attrs.push_back(attr);
    }
  }

  attrs.push_back(
    rewriter.getNamedAttr("multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
    rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(rshift)));


  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::SubConstOp>(op, newType, op.getInput(),
                                               attrs);
}

void SubConstLowering::LoweringINT4(PatternRewriter &rewriter,
                                    top::SubConstOp op, bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void SubConstLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::SubConstOp op) const {
  lowering_common_bf16<tpu::SubConstOp>(rewriter, op);
}

void SubConstLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::SubConstOp op) const {
  lowering_common_f16<tpu::SubConstOp>(rewriter, op);
}

void SubConstLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::SubConstOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir

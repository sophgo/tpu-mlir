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

void CompareLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::CompareOp compareOp) const {
  lowering_common_f32<tpu::CompareOp>(rewriter, compareOp.getOperation());
}

void CompareLowering::LoweringINT8(PatternRewriter &rewriter, top::CompareOp op,
                                   bool asymmetric) const {
  auto op_ = op.getOperation();
  double l_scale, r_scale;
  int64_t l_zp, r_zp;
  bool l_sign, r_sign;
  Quant::getScaleAndZeroPoint(op.lhs(), l_scale, l_zp, l_sign, asymmetric);
  Quant::getScaleAndZeroPoint(op.rhs(), r_scale, r_zp, r_sign, asymmetric);
  if (l_scale != r_scale || l_zp != r_zp || l_sign != r_sign) {
    lowering_common_f32<tpu::CompareOp>(rewriter, op_);
    return;
  }

  auto newType = Quant::getQuantBoolType(op.output());
  lowering_common<tpu::CompareOp>(rewriter, op_, newType);
}

void CompareLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::CompareOp compareOp) const {
  lowering_common_bf16<tpu::CompareOp>(rewriter, compareOp.getOperation());
}

void CompareLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::CompareOp compareOp) const {
  lowering_common_f16<tpu::CompareOp>(rewriter, compareOp.getOperation());
}

void CompareLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::CompareOp compareOp) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir

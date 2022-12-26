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
                                  top::CompareOp op) const {
  lowering_common_f32<tpu::CompareOp>(rewriter, op);
}

void CompareLowering::LoweringINT8(PatternRewriter &rewriter, top::CompareOp op,
                                   bool asymmetric) const {
  auto op_ = op.getOperation();
  double l_scale, r_scale;
  int64_t l_zp, r_zp;
  Quant::getScaleAndZeroPoint(op.lhs(), l_scale, l_zp, asymmetric);
  Quant::getScaleAndZeroPoint(op.rhs(), r_scale, r_zp, asymmetric);
  if (l_scale != r_scale || l_zp != r_zp) {
    lowering_common_f32<tpu::CompareOp>(rewriter, op_);
    return;
  }

  auto newType = getQuantBoolType(op.output());
  lowering_common<tpu::CompareOp>(rewriter, op_, newType);
}

void CompareLowering::LoweringINT4(PatternRewriter &rewriter, top::CompareOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
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

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

void ModLowering::LoweringF32(PatternRewriter &rewriter, top::ModOp op) const {
  lowering_common_f32<tpu::ModOp>(rewriter, op);
}

void ModLowering::LoweringINT8(PatternRewriter &rewriter, top::ModOp op,
                               bool asymmetric) const {
  lowering_common_f32<tpu::ModOp>(rewriter, op);
}
void ModLowering::LoweringINT4(PatternRewriter &rewriter, top::ModOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void ModLowering::LoweringBF16(PatternRewriter &rewriter, top::ModOp op) const {
  if(module::isBM1688()){
    lowering_common_bf16<tpu::ModOp>(rewriter, op);
  } else {
    lowering_common_f32<tpu::ModOp>(rewriter, op);
  }
}

void ModLowering::LoweringF16(PatternRewriter &rewriter, top::ModOp op) const {
  if(module::isBM1688()){
    lowering_common_f16<tpu::ModOp>(rewriter, op);
  } else {
    lowering_common_f32<tpu::ModOp>(rewriter, op);
  }
}

void ModLowering::LoweringF8(PatternRewriter &rewriter, top::ModOp op) const {
  llvm_unreachable("Not Implemented");
}

void ModLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::ModOp op) const {
  lowering_common<tpu::ModOp>(rewriter, op.getOperation(),
                              op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir

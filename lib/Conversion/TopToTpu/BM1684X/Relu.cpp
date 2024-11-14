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

void ReluLowering::LoweringF32(PatternRewriter &rewriter,
                               top::ReluOp op) const {
  lowering_common_f32<tpu::ReluOp>(rewriter, op);
}
void ReluLowering::LoweringINT4(PatternRewriter &rewriter, top::ReluOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void ReluLowering::LoweringINT8(PatternRewriter &rewriter, top::ReluOp op,
                                bool asymmetric) const {
  if (!asymmetric) {
    lowering_common_int8<tpu::ReluOp>(rewriter, op, asymmetric);
  } else {
    LoweringF32(rewriter, op);
  }
}

void ReluLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::ReluOp op) const {
  lowering_common_bf16<tpu::ReluOp>(rewriter, op);
}

void ReluLowering::LoweringF16(PatternRewriter &rewriter,
                               top::ReluOp op) const {
  lowering_common_f16<tpu::ReluOp>(rewriter, op);
}

void ReluLowering::LoweringF8(PatternRewriter &rewriter, top::ReluOp op) const {
  bool isE4 = module::getMode() == module::Mode::F8E4M3;
  lowering_common_f8<tpu::ReluOp>(rewriter, op, isE4);
}

void ReluLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::ReluOp op) const {
  lowering_common<tpu::ReluOp>(rewriter, op, op->getResult(0).getType());
}

} // namespace bm1684x
} // namespace tpu_mlir

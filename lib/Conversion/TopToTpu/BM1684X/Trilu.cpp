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

void TriluLowering::LoweringF32(PatternRewriter &rewriter,
                                top::TriluOp op) const {
  lowering_common_f32<tpu::TriluOp>(rewriter, op);
}
void TriluLowering::LoweringINT4(PatternRewriter &rewriter, top::TriluOp op,
                                 bool asymmetric) const {
  LoweringF32(rewriter, op);
}
void TriluLowering::LoweringINT8(PatternRewriter &rewriter, top::TriluOp op,
                                 bool asymmetric) const {
  // nodechip fix8b to be implemented,
  if (module::isMARS3() || module::isSGTPUV8())
    lowering_common_int8<tpu::TriluOp>(rewriter, op, asymmetric);
  else
    LoweringF32(rewriter, op);
}

void TriluLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::TriluOp op) const {
  if (module::isMARS3() || module::isSGTPUV8())
    lowering_common_bf16<tpu::TriluOp>(rewriter, op);
  else
    LoweringF32(rewriter, op);
}

void TriluLowering::LoweringF16(PatternRewriter &rewriter,
                                top::TriluOp op) const {
  LoweringF32(rewriter, op);
}

void TriluLowering::LoweringF8(PatternRewriter &rewriter,
                               top::TriluOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void TriluLowering::LoweringQuantized(PatternRewriter &rewriter,
                                      top::TriluOp op) const {
  LoweringF32(rewriter, op);
}

} // namespace bm1684x
} // namespace tpu_mlir

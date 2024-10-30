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

void MaskedFillLowering::LoweringF32(PatternRewriter &rewriter,
                                     top::MaskedFillOp op) const {
  lowering_common_f32<tpu::MaskedFillOp>(rewriter, op);
}

void MaskedFillLowering::LoweringINT8(PatternRewriter &rewriter, top::MaskedFillOp op,
                                      bool asymmetric) const {
  if(module::isMARS3())
    lowering_common_bf16<tpu::MaskedFillOp>(rewriter, op);
  else
    lowering_common_f32<tpu::MaskedFillOp>(rewriter, op);
}
void MaskedFillLowering::LoweringINT4(PatternRewriter &rewriter, top::MaskedFillOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void MaskedFillLowering::LoweringBF16(PatternRewriter &rewriter,
                                      top::MaskedFillOp op) const {
  lowering_common_bf16<tpu::MaskedFillOp>(rewriter, op);
}

void MaskedFillLowering::LoweringF16(PatternRewriter &rewriter,
                                     top::MaskedFillOp op) const {
  lowering_common_f16<tpu::MaskedFillOp>(rewriter, op);
}

void MaskedFillLowering::LoweringF8(PatternRewriter &rewriter,
                                     top::MaskedFillOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void MaskedFillLowering::LoweringQuantized(PatternRewriter &rewriter,
                                           top::MaskedFillOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir

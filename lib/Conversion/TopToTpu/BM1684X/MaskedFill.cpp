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
  lowering_common_f32<tpu::MaskedFillOp>(rewriter, op);
}

void MaskedFillLowering::LoweringBF16(PatternRewriter &rewriter,
                                      top::MaskedFillOp op) const {
  lowering_common_bf16<tpu::MaskedFillOp>(rewriter, op);
}

void MaskedFillLowering::LoweringF16(PatternRewriter &rewriter,
                                     top::MaskedFillOp op) const {
  lowering_common_f16<tpu::MaskedFillOp>(rewriter, op);
}

void MaskedFillLowering::LoweringQuantized(PatternRewriter &rewriter,
                                           top::MaskedFillOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir

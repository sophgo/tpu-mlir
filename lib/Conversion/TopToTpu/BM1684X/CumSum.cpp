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

void CumSumLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::CumSumOp op) const {
  lowering_common_f32<tpu::CumSumOp>(rewriter, op);
}

void CumSumLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::CumSumOp op) const {
  lowering_common_bf16<tpu::CumSumOp>(rewriter, op);
}

void CumSumLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::CumSumOp op) const {
  LoweringF32(rewriter, op);
}

void CumSumLowering::LoweringINT8(PatternRewriter &rewriter, top::CumSumOp op,
                                  bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void CumSumLowering::LoweringINT4(PatternRewriter &rewriter, top::CumSumOp op,
                                  bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void CumSumLowering::LoweringF8(PatternRewriter &rewriter,
                                top::CumSumOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void CumSumLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::CumSumOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir

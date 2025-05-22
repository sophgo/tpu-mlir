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

void CorrelationLowering::LoweringF32(PatternRewriter &rewriter,
                                      top::CorrelationOp op) const {
  lowering_common_f32<tpu::CorrelationOp>(rewriter, op);
}
void CorrelationLowering::LoweringINT4(PatternRewriter &rewriter,
                                       top::CorrelationOp op,
                                       bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void CorrelationLowering::LoweringINT8(PatternRewriter &rewriter,
                                       top::CorrelationOp op,
                                       bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void CorrelationLowering::LoweringBF16(PatternRewriter &rewriter,
                                       top::CorrelationOp op) const {
  lowering_common_bf16<tpu::CorrelationOp>(rewriter, op);
}

void CorrelationLowering::LoweringF16(PatternRewriter &rewriter,
                                      top::CorrelationOp op) const {
  lowering_common_f16<tpu::CorrelationOp>(rewriter, op);
}

void CorrelationLowering::LoweringF8(PatternRewriter &rewriter,
                                     top::CorrelationOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void CorrelationLowering::LoweringQuantized(PatternRewriter &rewriter,
                                            top::CorrelationOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir

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

void UpsampleLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::UpsampleOp op) const {
  lowering_common_f32<tpu::UpsampleOp>(rewriter, op);
}
void UpsampleLowering::LoweringINT4(PatternRewriter &rewriter,
                                    top::UpsampleOp op, bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void UpsampleLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::UpsampleOp op, bool asymmetric) const {
  lowering_common_int8<tpu::UpsampleOp>(rewriter, op, asymmetric);
}

void UpsampleLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::UpsampleOp op) const {
  lowering_common_bf16<tpu::UpsampleOp>(rewriter, op);
}

void UpsampleLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::UpsampleOp op) const {
  lowering_common_f16<tpu::UpsampleOp>(rewriter, op);
}

void UpsampleLowering::LoweringF8(PatternRewriter &rewriter,
                                  top::UpsampleOp op) const {
  lowering_common_f16<tpu::UpsampleOp>(rewriter, op);
}

void UpsampleLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::UpsampleOp op) const {
  lowering_common<tpu::UpsampleOp>(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir

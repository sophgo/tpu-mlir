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
  lowering_common_float<tpu::UpsampleOp>(rewriter, op);
}

void UpsampleLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::UpsampleOp op, bool asymmetric) const {
  lowering_common_int8<tpu::UpsampleOp>(rewriter, op, asymmetric);
}

void UpsampleLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::UpsampleOp op) const {
  lowering_common_float<tpu::UpsampleOp, BFloat16Type>(rewriter, op);
}

void UpsampleLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::UpsampleOp op) const {
  lowering_common_float<tpu::UpsampleOp, Float16Type>(rewriter, op);
}

void UpsampleLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::UpsampleOp op) const {
  lowering_common<tpu::UpsampleOp>(rewriter, op, op.output().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir

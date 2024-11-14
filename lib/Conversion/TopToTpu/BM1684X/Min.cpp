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

void MinLowering::LoweringF32(PatternRewriter &rewriter, top::MinOp op) const {
  lowering_common_f32<tpu::MinOp>(rewriter, op);
}
void MinLowering::LoweringINT4(PatternRewriter &rewriter, top::MinOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void MinLowering::LoweringINT8(PatternRewriter &rewriter, top::MinOp op,
                               bool asymmetric) const {
  lowering_common_int8<tpu::MinOp>(rewriter, op, asymmetric);
}

void MinLowering::LoweringBF16(PatternRewriter &rewriter, top::MinOp op) const {
  lowering_common_bf16<tpu::MinOp>(rewriter, op);
}

void MinLowering::LoweringF16(PatternRewriter &rewriter, top::MinOp op) const {
  lowering_common_f16<tpu::MinOp>(rewriter, op);
}

void MinLowering::LoweringF8(PatternRewriter &rewriter, top::MinOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void MinLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::MinOp op) const {
  lowering_common<tpu::MinOp>(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir

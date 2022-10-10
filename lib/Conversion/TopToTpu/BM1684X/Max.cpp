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

void MaxLowering::LoweringF32(PatternRewriter &rewriter, top::MaxOp op) const {
  lowering_common_float<tpu::MaxOp, Float32Type>(rewriter, op);
}

void MaxLowering::LoweringINT8(PatternRewriter &rewriter, top::MaxOp op,
                               bool asymmetric) const {
  lowering_common_int8<tpu::MaxOp>(rewriter, op, asymmetric);
}

void MaxLowering::LoweringBF16(PatternRewriter &rewriter, top::MaxOp op) const {
  lowering_common_float<tpu::MaxOp, BFloat16Type>(rewriter, op);
}

void MaxLowering::LoweringF16(PatternRewriter &rewriter, top::MaxOp op) const {
  lowering_common_float<tpu::MaxOp, Float16Type>(rewriter, op);
}

void MaxLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::MaxOp op) const {
  lowering_common<tpu::MaxOp>(rewriter, op, op.output().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir

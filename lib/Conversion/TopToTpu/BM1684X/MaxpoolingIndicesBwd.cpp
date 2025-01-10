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

void MaxPoolingIndicesBwdLowering::LoweringF32(
    PatternRewriter &rewriter, top::MaxPoolingIndicesBwdOp op) const {
  lowering_common_f32<tpu::MaxPoolingIndicesBwdOp>(rewriter, op);
}

void MaxPoolingIndicesBwdLowering::LoweringF16(
    PatternRewriter &rewriter, top::MaxPoolingIndicesBwdOp op) const {
  lowering_common_f16<tpu::MaxPoolingIndicesBwdOp>(rewriter, op);
}

void MaxPoolingIndicesBwdLowering::LoweringBF16(
    PatternRewriter &rewriter, top::MaxPoolingIndicesBwdOp op) const {
  lowering_common_bf16<tpu::MaxPoolingIndicesBwdOp>(rewriter, op);
}

void MaxPoolingIndicesBwdLowering::LoweringINT8(PatternRewriter &rewriter,
                                                top::MaxPoolingIndicesBwdOp op,
                                                bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void MaxPoolingIndicesBwdLowering::LoweringINT4(PatternRewriter &rewriter,
                                                top::MaxPoolingIndicesBwdOp op,
                                                bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void MaxPoolingIndicesBwdLowering::LoweringQuantized(
    PatternRewriter &rewriter, top::MaxPoolingIndicesBwdOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void MaxPoolingIndicesBwdLowering::LoweringF8(
    PatternRewriter &rewriter, top::MaxPoolingIndicesBwdOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
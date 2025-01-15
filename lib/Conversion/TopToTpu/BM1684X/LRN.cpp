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

void LRNLowering::LoweringF32(PatternRewriter &rewriter, top::LRNOp op) const {
  lowering_common_f32<tpu::LRNOp>(rewriter, op, 3);
}

void LRNLowering::LoweringINT4(PatternRewriter &rewriter, top::LRNOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void LRNLowering::LoweringINT8(PatternRewriter &rewriter, top::LRNOp LRNOp,
                               bool asymmetric) const {
  if (module::isMARS3() || module::isSGTPUV8()) {
    // lowering_common_int8<tpu::LRNOp>(rewriter, LRNOp, asymmetric, 3);
    LoweringBF16(rewriter, LRNOp);
  } else
    LoweringF32(rewriter, LRNOp);
}

void LRNLowering::LoweringBF16(PatternRewriter &rewriter, top::LRNOp op) const {
  if (module::isMARS3() || module::isSGTPUV8())
    lowering_common_bf16<tpu::LRNOp>(rewriter, op, 3);
  else
    LoweringF32(rewriter, op);
}

void LRNLowering::LoweringF16(PatternRewriter &rewriter, top::LRNOp op) const {
  LoweringF32(rewriter, op);
}

void LRNLowering::LoweringF8(PatternRewriter &rewriter, top::LRNOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void LRNLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::LRNOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir

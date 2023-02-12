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

void WhereLowering::LoweringF32(PatternRewriter &rewriter,
                                top::WhereOp op) const {
  lowering_common_f32<tpu::WhereOp>(rewriter, op);
}
void WhereLowering::LoweringINT4(PatternRewriter &rewriter, top::WhereOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void WhereLowering::LoweringINT8(PatternRewriter &rewriter, top::WhereOp op,
                               bool asymmetric) const {
  lowering_common_int8<tpu::WhereOp>(rewriter, op);
}

void WhereLowering::LoweringBF16(PatternRewriter &rewriter,
                               top::WhereOp op) const {
  lowering_common_bf16<tpu::WhereOp>(rewriter, op);
}

void WhereLowering::LoweringF16(PatternRewriter &rewriter,
                              top::WhereOp op) const {
  lowering_common_f16<tpu::WhereOp>(rewriter, op);
}

void WhereLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::WhereOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir

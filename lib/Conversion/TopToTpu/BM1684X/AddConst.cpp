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

void AddConstLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::AddConstOp op) const {
  lowering_common_f32<tpu::AddConstOp>(rewriter, op);
}

void AddConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::AddConstOp op, bool asymmetric) const {
  lowering_common_f32<tpu::AddConstOp>(rewriter, op);
}

void AddConstLowering::LoweringINT4(PatternRewriter &rewriter, top::AddConstOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void AddConstLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::AddConstOp op) const {
  lowering_common_bf16<tpu::AddConstOp>(rewriter, op);
}

void AddConstLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::AddConstOp op) const {
  lowering_common_f16<tpu::AddConstOp>(rewriter, op);
}

void AddConstLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::AddConstOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

void top::MaxPoolWithMaskOp::lowering_int8_bm1684x(PatternRewriter &rewriter,
                                                   bool asymmetric) {
  lowering_f32_bm1684x(rewriter);
}

void top::MaxPoolWithMaskOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  auto op = getOperation();
  rewriter.replaceOpWithNewOp<tpu::MaxPoolWithMaskOp>(
      op, op->getResultTypes(), op->getOperands(), op->getAttrs());
}

void top::MaxPoolWithMaskOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("Not Implemented");
}

void top::MaxPoolWithMaskOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("Not Implemented");
}

void top::MaxPoolWithMaskOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("Not Implemented");
}

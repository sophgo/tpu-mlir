//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

void top::AbsOp::lowering_int8_bm1684x(PatternRewriter &rewriter, bool asymmetric) {
  lowering_common_int8<tpu::AbsOp>(rewriter, getOperation(), asymmetric);
}

void top::AbsOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::AbsOp, Float32Type>(rewriter, getOperation());
}

void top::AbsOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::AbsOp, BFloat16Type>(rewriter, getOperation());
}

void top::AbsOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::AbsOp, Float16Type>(rewriter, getOperation());
}

void top::AbsOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("Not Implemented");
}

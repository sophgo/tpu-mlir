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

void top::ReluOp::lowering_int8_bm1684x(PatternRewriter &rewriter,
                                        bool asymmetric) {
  lowering_common_int8<tpu::ReluOp>(rewriter, getOperation(), asymmetric);
}

void top::ReluOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::ReluOp>(rewriter, getOperation());
}

void top::ReluOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::ReluOp, BFloat16Type>(rewriter, getOperation());
}

void top::ReluOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::ReluOp, Float16Type>(rewriter, getOperation());
}

void top::ReluOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("not support now");
}

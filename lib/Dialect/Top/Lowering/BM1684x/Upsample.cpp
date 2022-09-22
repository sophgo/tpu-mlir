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

void top::UpsampleOp::lowering_int8_bm1684x(PatternRewriter &rewriter,
                                            bool asymmetric) {
  lowering_common_int8<tpu::UpsampleOp>(rewriter, getOperation(), asymmetric);
}

void top::UpsampleOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::UpsampleOp>(rewriter, getOperation());
}

void top::UpsampleOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::UpsampleOp, BFloat16Type>(rewriter,
                                                       getOperation());
}

void top::UpsampleOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::UpsampleOp, Float16Type>(rewriter, getOperation());
}

void top::UpsampleOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  lowering_common<tpu::UpsampleOp>(rewriter, getOperation(),
                                   output().getType());
}

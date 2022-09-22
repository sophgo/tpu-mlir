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

void top::TileOp::lowering_int8_bm1684x(PatternRewriter &rewriter,
                                        bool asymmetric) {
  lowering_common_int8<tpu::TileOp>(rewriter, getOperation(), asymmetric);
}

void top::TileOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::TileOp>(rewriter, getOperation());
}

void top::TileOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::TileOp, BFloat16Type>(rewriter, getOperation());
}

void top::TileOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::TileOp, Float16Type>(rewriter, getOperation());
}

void top::TileOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  lowering_common<tpu::TileOp>(rewriter, getOperation(), output().getType());
}

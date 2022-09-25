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

void top::MaxUnpoolOp::lowering_int8_bm1684x(PatternRewriter &rewriter,
                                            bool asymmetric) {
  lowering_f32_bm1684x(rewriter);
}

void top::MaxUnpoolOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::MaxUnpoolOp>(rewriter, getOperation());
}

void top::MaxUnpoolOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("Not Implemented");
}

void top::MaxUnpoolOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("Not Implemented");
}

void top::MaxUnpoolOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("Not Implemented");
}

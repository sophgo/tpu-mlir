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

void top::BatchNormOp::lowering_int8_bm1684x(PatternRewriter &rewriter, bool asymmetric) {
  llvm_unreachable("BatchNormOp to be supported");
}

void top::BatchNormOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("BatchNormOp to be supported");
}

void top::BatchNormOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("BatchNormOp to be supported");
}

void top::BatchNormOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("BatchNormOp to be supported");
}

void top::BatchNormOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("not support now");
}

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

void top::ScaleOp::lowering_int8_bm1684(PatternRewriter &rewriter) {
  llvm_unreachable("Not Implemented");
}

void top::ScaleOp::lowering_f32_bm1684(PatternRewriter &rewriter) {
  lowering_common_float<tpu::ScaleOp>(rewriter, getOperation());
}

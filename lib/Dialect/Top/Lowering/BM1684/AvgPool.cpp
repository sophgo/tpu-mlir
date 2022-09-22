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
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

void top::AvgPoolOp::lowering_int8_bm1684(PatternRewriter &rewriter) {
  auto op = getOperation();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Avg));
  if (kernel_shape().size() == 3) {
    lowering_common_int8<tpu::Pool3DOp>(rewriter, op);
  } else if (kernel_shape().size() == 2) {
    lowering_common_int8<tpu::Pool2DOp>(rewriter, op);
  } else {
    lowering_common_int8<tpu::Pool1DOp>(rewriter, op);
  }
}

void top::AvgPoolOp::lowering_f32_bm1684(PatternRewriter &rewriter) {
  auto op = getOperation();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Avg));
  if (kernel_shape().size() == 3) {
    lowering_common_float<tpu::Pool3DOp>(rewriter, op);
  } else if (kernel_shape().size() == 2) {
    lowering_common_float<tpu::Pool2DOp>(rewriter, op);
  } else {
    lowering_common_float<tpu::Pool1DOp>(rewriter, op);
  }
}

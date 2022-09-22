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

void top::MaxPoolOp::lowering_int8_bm1684x(PatternRewriter &rewriter,
                                           bool asymmetric) {
  auto op = getOperation();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  if (kernel_shape().size() == 3) {
    lowering_common_int8<tpu::Pool3DOp>(rewriter, op, asymmetric);
  } else if (kernel_shape().size() == 2) {
    lowering_common_int8<tpu::Pool2DOp>(rewriter, op, asymmetric);
  } else {
    lowering_common_int8<tpu::Pool1DOp>(rewriter, op, asymmetric);
  }
}

void top::MaxPoolOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  auto op = getOperation();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  if (kernel_shape().size() == 3) {
    lowering_common_float<tpu::Pool3DOp>(rewriter, op);
  } else if (kernel_shape().size() == 2) {
    lowering_common_float<tpu::Pool2DOp>(rewriter, op);
  } else {
    lowering_common_float<tpu::Pool1DOp>(rewriter, op);
  }
}

void top::MaxPoolOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  auto op = getOperation();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  if (kernel_shape().size() == 3) {
    lowering_common_float<tpu::Pool3DOp, BFloat16Type>(rewriter, op);
  } else if (kernel_shape().size() == 2) {
    lowering_common_float<tpu::Pool2DOp, BFloat16Type>(rewriter, op);
  } else {
    lowering_common_float<tpu::Pool1DOp, BFloat16Type>(rewriter, op);
  }
}

void top::MaxPoolOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  auto op = getOperation();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  if (kernel_shape().size() == 3) {
    lowering_common_float<tpu::Pool3DOp, Float16Type>(rewriter, op);
  } else if (kernel_shape().size() == 2) {
    lowering_common_float<tpu::Pool2DOp, Float16Type>(rewriter, op);
  } else {
    lowering_common_float<tpu::Pool1DOp, Float16Type>(rewriter, op);
  }
}

void top::MaxPoolOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  auto op = getOperation();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  if (kernel_shape().size() == 3) {
    lowering_common<tpu::Pool3DOp>(rewriter, op, output().getType());
  } else if (kernel_shape().size() == 2) {
    lowering_common<tpu::Pool2DOp>(rewriter, op, output().getType());
  } else {
    lowering_common<tpu::Pool1DOp>(rewriter, op, output().getType());
  }
}

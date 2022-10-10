//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

void MaxPoolLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::MaxPoolOp op) const {
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  if (op.kernel_shape().size() == 3) {
    lowering_common_float<tpu::Pool3DOp>(rewriter, op);
  } else if (op.kernel_shape().size() == 2) {
    lowering_common_float<tpu::Pool2DOp>(rewriter, op);
  } else {
    lowering_common_float<tpu::Pool1DOp>(rewriter, op);
  }
}

void MaxPoolLowering::LoweringINT8(PatternRewriter &rewriter, top::MaxPoolOp op,
                                   bool asymmetric) const {
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  if (op.kernel_shape().size() == 3) {
    lowering_common_int8<tpu::Pool3DOp>(rewriter, op, asymmetric);
  } else if (op.kernel_shape().size() == 2) {
    lowering_common_int8<tpu::Pool2DOp>(rewriter, op, asymmetric);
  } else {
    lowering_common_int8<tpu::Pool1DOp>(rewriter, op, asymmetric);
  }
}

void MaxPoolLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::MaxPoolOp op) const {
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  if (op.kernel_shape().size() == 3) {
    lowering_common_float<tpu::Pool3DOp, BFloat16Type>(rewriter, op);
  } else if (op.kernel_shape().size() == 2) {
    lowering_common_float<tpu::Pool2DOp, BFloat16Type>(rewriter, op);
  } else {
    lowering_common_float<tpu::Pool1DOp, BFloat16Type>(rewriter, op);
  }
}

void MaxPoolLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::MaxPoolOp op) const {
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  if (op.kernel_shape().size() == 3) {
    lowering_common_float<tpu::Pool3DOp, Float16Type>(rewriter, op);
  } else if (op.kernel_shape().size() == 2) {
    lowering_common_float<tpu::Pool2DOp, Float16Type>(rewriter, op);
  } else {
    lowering_common_float<tpu::Pool1DOp, Float16Type>(rewriter, op);
  }
}

void MaxPoolLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::MaxPoolOp op) const {
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  if (op.kernel_shape().size() == 3) {
    lowering_common<tpu::Pool3DOp>(rewriter, op, op.output().getType());
  } else if (op.kernel_shape().size() == 2) {
    lowering_common<tpu::Pool2DOp>(rewriter, op, op.output().getType());
  } else {
    lowering_common<tpu::Pool1DOp>(rewriter, op, op.output().getType());
  }
}

} // namespace bm1684x
} // namespace tpu_mlir

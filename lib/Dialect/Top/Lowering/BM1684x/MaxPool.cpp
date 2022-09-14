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

Value top::MaxPoolOp::lowering_int8_bm1684x(bool asymmetric) {
  Value newValue;
  if (kernel_shape().size() == 3) {
    newValue =
        lowering_common_int8<tpu::Pool3DOp>(getOperation(), asymmetric);
  } else if (kernel_shape().size() == 2) {
    newValue =
        lowering_common_int8<tpu::Pool2DOp>(getOperation(), asymmetric);
  } else {
    newValue =
        lowering_common_int8<tpu::Pool1DOp>(getOperation(), asymmetric);
  }
  auto op = newValue.getDefiningOp();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  return newValue;
}

Value top::MaxPoolOp::lowering_f32_bm1684x() {
  Value newValue;
  if (kernel_shape().size() == 3) {
    newValue = lowering_common_float<tpu::Pool3DOp>(getOperation());
  } else if (kernel_shape().size() == 2) {
    newValue = lowering_common_float<tpu::Pool2DOp>(getOperation());
  } else {
    newValue = lowering_common_float<tpu::Pool1DOp>(getOperation());
  }
  auto op = newValue.getDefiningOp();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  return newValue;
}

Value top::MaxPoolOp::lowering_bf16_bm1684x() {
  Value newValue;
  if (kernel_shape().size() == 3) {
    newValue =
        lowering_common_float<tpu::Pool3DOp, BFloat16Type>(getOperation());
  } else if (kernel_shape().size() == 2) {
    newValue =
        lowering_common_float<tpu::Pool2DOp, BFloat16Type>(getOperation());
  } else {
    newValue =
        lowering_common_float<tpu::Pool1DOp, BFloat16Type>(getOperation());
  }
  auto op = newValue.getDefiningOp();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  return newValue;
}

Value top::MaxPoolOp::lowering_f16_bm1684x() {
  Value newValue;
  if (kernel_shape().size() == 3) {
    newValue =
        lowering_common_float<tpu::Pool3DOp, Float16Type>(getOperation());
  } else if (kernel_shape().size() == 2) {
    newValue =
        lowering_common_float<tpu::Pool2DOp, Float16Type>(getOperation());
  } else {
    newValue =
        lowering_common_float<tpu::Pool1DOp, Float16Type>(getOperation());
  }
  auto op = newValue.getDefiningOp();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  return newValue;
}

Value top::MaxPoolOp::lowering_quant_bm1684x() {
  Value newValue;
  if (kernel_shape().size() == 3) {
    newValue =
        lowering_common<tpu::Pool3DOp>(getOperation(), output().getType());
  } else if (kernel_shape().size() == 2) {
    newValue =
        lowering_common<tpu::Pool2DOp>(getOperation(), output().getType());

  } else {
    newValue =
        lowering_common<tpu::Pool1DOp>(getOperation(), output().getType());
  }
  auto op = newValue.getDefiningOp();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  return newValue;
}

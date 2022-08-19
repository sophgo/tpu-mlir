//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
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

Value top::MaxPoolOp::lowering_int8_bm1684x(bool asymmetric) {
  Value newValue;
  if (kernel_shape().size() == 3) {
    newValue =
        lowering_common_int8<tpu::MaxPool3DOp>(getOperation(), asymmetric);
  } else if (kernel_shape().size() == 2) {
    newValue =
        lowering_common_int8<tpu::MaxPool2DOp>(getOperation(), asymmetric);
  } else {
    newValue =
        lowering_common_int8<tpu::MaxPool1DOp>(getOperation(), asymmetric);
  }
  return newValue;
}

Value top::MaxPoolOp::lowering_f32_bm1684x() {
  Value newValue;
  if (kernel_shape().size() == 3) {
    newValue = lowering_common_float<tpu::MaxPool3DOp>(getOperation());
  } else if (kernel_shape().size() == 2) {
    newValue = lowering_common_float<tpu::MaxPool2DOp>(getOperation());
  } else {
    newValue = lowering_common_float<tpu::MaxPool1DOp>(getOperation());
  }
  return newValue;
}

Value top::MaxPoolOp::lowering_bf16_bm1684x() {
  Value newValue;
  if (kernel_shape().size() == 3) {
    newValue =
        lowering_common_float<tpu::MaxPool3DOp, BFloat16Type>(getOperation());
  } else if (kernel_shape().size() == 2) {
    newValue =
        lowering_common_float<tpu::MaxPool2DOp, BFloat16Type>(getOperation());
  } else {
    newValue =
        lowering_common_float<tpu::MaxPool1DOp, BFloat16Type>(getOperation());
  }
  return newValue;
}

Value top::MaxPoolOp::lowering_f16_bm1684x() {
  Value newValue;
  if (kernel_shape().size() == 3) {
    newValue =
        lowering_common_float<tpu::MaxPool3DOp, Float16Type>(getOperation());
  } else if (kernel_shape().size() == 2) {
    newValue =
        lowering_common_float<tpu::MaxPool2DOp, Float16Type>(getOperation());
  } else {
    newValue =
        lowering_common_float<tpu::MaxPool1DOp, Float16Type>(getOperation());
  }
  return newValue;
}

Value top::MaxPoolOp::lowering_quant_bm1684x() {
  Value newValue;
  if (kernel_shape().size() == 3) {
    newValue =
        lowering_common<tpu::MaxPool3DOp>(getOperation(), output().getType());
  } else if (kernel_shape().size() == 2) {
    newValue =
        lowering_common<tpu::MaxPool2DOp>(getOperation(), output().getType());

  } else {
    newValue =
        lowering_common<tpu::MaxPool1DOp>(getOperation(), output().getType());
  }
  return newValue;
}

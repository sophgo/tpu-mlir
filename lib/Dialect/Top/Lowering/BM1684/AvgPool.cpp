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

Value top::AvgPoolOp::lowering_int8_bm1684() {
  Value newValue;
  if (kernel_shape().size() == 3) {
    newValue = lowering_common_int8<tpu::AvgPool3DOp>(getOperation());
  } else if (kernel_shape().size() == 2) {
    newValue = lowering_common_int8<tpu::AvgPool2DOp>(getOperation());
  } else {
    newValue = lowering_common_int8<tpu::AvgPool1DOp>(getOperation());
  }
  return newValue;
}

Value top::AvgPoolOp::lowering_f32_bm1684() {
  Value newValue;
  if (kernel_shape().size() == 3) {
    newValue = lowering_common_float<tpu::AvgPool3DOp>(getOperation());
  } else if (kernel_shape().size() == 2) {
    newValue = lowering_common_float<tpu::AvgPool2DOp>(getOperation());
  } else {
    newValue = lowering_common_float<tpu::AvgPool1DOp>(getOperation());
  }
  return newValue;
}

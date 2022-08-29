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

Value top::MaxPoolOp::lowering_int8_bm1684() {
  Value newValue;
  if (kernel_shape().size() == 3) {
    newValue = lowering_common_int8<tpu::MaxPool3DOp>(getOperation());
  } else if (kernel_shape().size() == 2) {
    newValue = lowering_common_int8<tpu::MaxPool2DOp>(getOperation());
  } else {
    newValue = lowering_common_int8<tpu::MaxPool1DOp>(getOperation());
  }
  return newValue;
}

Value top::MaxPoolOp::lowering_f32_bm1684() {
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

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
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

Value top::AvgPoolOp::lowering_int8_bm1684x(bool asymetric) {
  return lowering_common_int8<tpu::AvgPoolOp>(getOperation(), asymetric);
}

Value top::AvgPoolOp::lowering_f32_bm1684x() {
  return lowering_common<tpu::AvgPoolOp>(getOperation());
}

Value top::AvgPoolOp::lowering_bf16_bm1684x() {
  return lowering_common<tpu::AvgPoolOp, BFloat16Type>(getOperation());
}

Value top::AvgPoolOp::lowering_f16_bm1684x() {
  return lowering_common<tpu::AvgPoolOp, Float16Type>(getOperation());
}

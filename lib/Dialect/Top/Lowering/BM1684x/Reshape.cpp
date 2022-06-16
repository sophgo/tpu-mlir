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

Value top::ReshapeOp::lowering_int8_bm1684x(bool asymetric) {
  return lowering_common_int8<tpu::ReshapeOp>(getOperation(), asymetric);
}


Value top::ReshapeOp::lowering_f32_bm1684x() {
  return lowering_common<tpu::ReshapeOp>(getOperation());
}

Value top::ReshapeOp::lowering_bf16_bm1684x() {
  return lowering_common<tpu::ReshapeOp, BFloat16Type>(getOperation());
}

Value top::ReshapeOp::lowering_f16_bm1684x() {
  return lowering_common<tpu::ReshapeOp, Float16Type>(getOperation());
}

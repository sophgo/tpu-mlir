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

Value top::ConcatOp::lowering_int8_bm1684x(bool asymmetric) {
  return lowering_common_int8<tpu::ConcatOp>(getOperation(), asymmetric);
}

Value top::ConcatOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::ConcatOp>(getOperation());
}

Value top::ConcatOp::lowering_bf16_bm1684x() {
  return lowering_common_float<tpu::ConcatOp, BFloat16Type>(getOperation());
}

Value top::ConcatOp::lowering_f16_bm1684x() {
  return lowering_common_float<tpu::ConcatOp, Float16Type>(getOperation());
}

Value top::ConcatOp::lowering_quant_bm1684x() {
  return lowering_common<tpu::ConcatOp>(getOperation(), output().getType());
}

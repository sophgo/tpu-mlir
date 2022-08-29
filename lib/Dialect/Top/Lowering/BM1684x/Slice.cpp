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
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

Value top::SliceOp::lowering_int8_bm1684x(bool asymmetric) {
  return lowering_common_int8<tpu::SliceOp>(getOperation(), asymmetric);
}

Value top::SliceOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::SliceOp>(getOperation());
}

Value top::SliceOp::lowering_bf16_bm1684x() {
  return lowering_common_float<tpu::SliceOp, BFloat16Type>(getOperation());
}

Value top::SliceOp::lowering_f16_bm1684x() {
  return lowering_common_float<tpu::SliceOp, Float16Type>(getOperation());
}

Value top::SliceOp::lowering_quant_bm1684x() {
  return lowering_common<tpu::SliceOp>(getOperation(), output().getType());
}

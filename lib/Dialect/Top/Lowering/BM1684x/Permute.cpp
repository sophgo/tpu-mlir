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
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

Value top::PermuteOp::lowering_int8_bm1684x(bool asymmetric) {
  return lowering_common_int8<tpu::PermuteOp>(getOperation(), asymmetric);
}

Value top::PermuteOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::PermuteOp>(getOperation());
}

Value top::PermuteOp::lowering_bf16_bm1684x() {
  return lowering_common_float<tpu::PermuteOp, BFloat16Type>(getOperation());
}

Value top::PermuteOp::lowering_f16_bm1684x() {
  return lowering_common_float<tpu::PermuteOp, Float16Type>(getOperation());
}

Value top::PermuteOp::lowering_quant_bm1684x() {
  return lowering_common<tpu::PermuteOp>(getOperation(), output().getType());
}

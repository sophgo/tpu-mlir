//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

Value top::MaxOp::lowering_int8_bm1684x(bool asymmetric) {
  return lowering_common_int8<tpu::MaxOp>(getOperation(), asymmetric);
}

Value top::MaxOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::MaxOp, Float32Type>(getOperation());
}

Value top::MaxOp::lowering_bf16_bm1684x() {
  return lowering_common_float<tpu::MaxOp, BFloat16Type>(getOperation());
}

Value top::MaxOp::lowering_f16_bm1684x() {
  return lowering_common_float<tpu::MaxOp, Float16Type>(getOperation());
}

Value top::MaxOp::lowering_quant_bm1684x() {
  return lowering_common<tpu::MaxOp>(getOperation(), output().getType());
}

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

Value top::PadOp::lowering_int8_bm1684x(bool asymmetric) {
  return lowering_common_int8<tpu::PadOp>(getOperation(), asymmetric);
}

Value top::PadOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::PadOp>(getOperation());
}

Value top::PadOp::lowering_bf16_bm1684x() {
  return lowering_common_float<tpu::PadOp, BFloat16Type>(getOperation());
}

Value top::PadOp::lowering_f16_bm1684x() {
  return lowering_common_float<tpu::PadOp, Float16Type>(getOperation());
}

Value top::PadOp::lowering_quant_bm1684x() {
  llvm_unreachable("not support now");
}

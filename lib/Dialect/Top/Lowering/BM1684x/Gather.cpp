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

Value top::GatherOp::lowering_int8_bm1684x(bool asymmetric) {
  return lowering_common_float<tpu::GatherOp>(getOperation());
}

Value top::GatherOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::GatherOp>(getOperation());
}

Value top::GatherOp::lowering_bf16_bm1684x() {
  llvm_unreachable("GatherOp to be supported");
  return nullptr;
}

Value top::GatherOp::lowering_f16_bm1684x() {
  llvm_unreachable("GatherOp to be supported");
  return nullptr;
}

Value top::GatherOp::lowering_quant_bm1684x() {
  llvm_unreachable("not support now");
}

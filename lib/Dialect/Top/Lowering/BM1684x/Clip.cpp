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

Value top::ClipOp::lowering_int8_bm1684x(bool asymmetric) {
  llvm_unreachable("ClipOp not support now");
}

Value top::ClipOp::lowering_f32_bm1684x() {
  llvm_unreachable("ClipOp not support now");
  // return lowering_common_float<tpu::ClipOp>(getOperation());
}

Value top::ClipOp::lowering_bf16_bm1684x() {
  llvm_unreachable("ClipOp not support now");
  // return lowering_common_float<tpu::ClipOp, BFloat16Type>(getOperation());
}

Value top::ClipOp::lowering_f16_bm1684x() {
  llvm_unreachable("ClipOp not support now");
  // return lowering_common_float<tpu::ClipOp, Float16Type>(getOperation());
}

Value top::ClipOp::lowering_quant_bm1684x() {
  llvm_unreachable("ClipOp not support now");
}

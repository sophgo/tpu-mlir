//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

Value top::BatchNormOp::lowering_int8_bm1684x(bool asymmetric) {
  llvm_unreachable("BatchNormOp to be supported");
  return nullptr;
}

Value top::BatchNormOp::lowering_f32_bm1684x() {
  llvm_unreachable("BatchNormOp to be supported");
  return nullptr;
}

Value top::BatchNormOp::lowering_bf16_bm1684x() {
  llvm_unreachable("BatchNormOp to be supported");
  return nullptr;
}

Value top::BatchNormOp::lowering_f16_bm1684x() {
  llvm_unreachable("BatchNormOp to be supported");
  return nullptr;
}

Value top::BatchNormOp::lowering_quant_bm1684x() {
  llvm_unreachable("not support now");
}

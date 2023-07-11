//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//


#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::CustomOp::init(InferenceParameter &p) { return success(); }
void tpu::CustomOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::CustomOp::inference(InferenceParameter &p) {
  llvm_unreachable("CustomOp no need to inference");
  return success();
}

mlir::Type tpu::CustomOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return type_verify_case_same(getOperation(), opd_idx, mode);
}


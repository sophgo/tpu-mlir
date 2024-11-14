//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::Device2HostOp::init(InferenceParameter &p) {
  return success();
}
void tpu::Device2HostOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::Device2HostOp::inference(InferenceParameter &p) {
  const auto bytes = sizeof(float) * module::getNumElements(getInput());
  memcpy(p.outputs[0], p.inputs[0], bytes);
  return success();
}

mlir::Type tpu::Device2HostOp::type_verify(uint64_t opd_idx,
                                           TypeCastMode &mode) {
  auto op = getOperation();
  return type_verify_case_same(op, 0, mode);
}

bool tpu::Device2HostOp::support_multi_core() { return false; }

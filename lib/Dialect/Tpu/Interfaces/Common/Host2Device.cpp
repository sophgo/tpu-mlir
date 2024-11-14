//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::Host2DeviceOp::init(InferenceParameter &p) {
  return success();
}
void tpu::Host2DeviceOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::Host2DeviceOp::inference(InferenceParameter &p) {
  const auto bytes = module::getBytes(getInput());
  memcpy(p.outputs[0], p.inputs[0], bytes);
  return success();
}

mlir::Type tpu::Host2DeviceOp::type_verify(uint64_t opd_idx,
                                           TypeCastMode &mode) {
  auto op = getOperation();
  return type_verify_case_same(op, 0, mode);
}

bool tpu::Host2DeviceOp::support_multi_core() { return false; }

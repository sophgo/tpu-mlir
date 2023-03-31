//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::Device2HostOp::init(InferenceParameter &p) { return success(); }
void tpu::Device2HostOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::Device2HostOp::inference(InferenceParameter &p) {
  const auto bytes = module::getBytes(getInput());
  memcpy(p.outputs[0], p.inputs[0], bytes);
  return success();
}

mlir::Type tpu::Device2HostOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return do_nothing(mode);
}

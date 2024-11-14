//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"

LogicalResult tpu::IdentityOp::init(InferenceParameter &p) { return success(); }

void tpu::IdentityOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::IdentityOp::inference(InferenceParameter &p) {
  return success();
}

bool tpu::IdentityOp::support_multi_core() { return false; }

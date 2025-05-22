//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::WhereBnbwdOp::init(InferenceParameter &p) {

  return success();
}

void tpu::WhereBnbwdOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::WhereBnbwdOp::inference(InferenceParameter &p) {
  return success();
}

bool tpu::WhereBnbwdOp::support_multi_core() { return true; }

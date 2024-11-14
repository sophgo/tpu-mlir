//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
using namespace tpu_mlir::backend;

LogicalResult tpu::LoadToL2MOp::init(InferenceParameter &p) {
  return success();
}
void tpu::LoadToL2MOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::LoadToL2MOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

bool tpu::LoadToL2MOp::support_multi_core() { return false; }

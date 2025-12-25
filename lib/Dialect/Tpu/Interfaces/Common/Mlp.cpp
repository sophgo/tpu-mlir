//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::MlpOp::init(InferenceParameter &p) { return success(); }

void tpu::MlpOp::deinit(InferenceParameter &p) { return; }

LogicalResult tpu::MlpOp::inference(InferenceParameter &p) { return success(); }

bool tpu::MlpOp::support_multi_core() {
  return (module::isSG2380() || module::isBM1690Family()) &&
         !module::isOpInGroupParallel(*this);
}

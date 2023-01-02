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

#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Float16.h"



LogicalResult tpu::CompareOp::init(InferenceParameter &p) { return success(); }
void tpu::CompareOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::CompareOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    p.outputs[0][i] = compare(p.inputs[0][i], p.inputs[1][i], getMode());
  }
  return success();
}

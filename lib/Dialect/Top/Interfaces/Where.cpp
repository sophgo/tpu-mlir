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
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"




int64_t top::WhereOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::WhereOp::init(InferenceParameter &p) { return success(); }
void top::WhereOp::deinit(InferenceParameter &p) {}

LogicalResult top::WhereOp::inference(InferenceParameter &p) {
  const auto num_element = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    p.outputs[0][i] = p.inputs[0][i] ? p.inputs[1][i] : p.inputs[2][i];
  }
  return success();
}

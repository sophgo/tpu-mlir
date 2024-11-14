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

int64_t top::ModOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::ModOp::init(InferenceParameter &p) { return success(); }

LogicalResult top::ModOp::inference(InferenceParameter &p) {
  // if (p.handle == nullptr) {
  //   return failure();
  // }
  auto num_element = module::getNumElements(getOutput());
  auto input2_size = module::getNumElements(getInputs()[1]);
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int64_t i = 0; i < num_element; i++) {
    double divisor = (input2_size == 1) ? p.inputs[1][0] : p.inputs[1][i];
    double mid = p.inputs[0][i] / divisor;
    p.outputs[0][i] = p.inputs[0][i] - static_cast<int>(mid) * divisor;
  }
  return success();
}

void top::ModOp::deinit(InferenceParameter &p) {}

void top::ModOp::shape_inference() {
  broadcast_shape_inference(getOperation());
}

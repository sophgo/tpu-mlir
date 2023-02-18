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


bool top::SqueezeOp::isEltwise() {
  return false;
}

int64_t top::SqueezeOp::getFLOPs() { return 0; }

LogicalResult top::SqueezeOp::init(InferenceParameter &p) { return success(); }
void top::SqueezeOp::deinit(InferenceParameter &p) {}

LogicalResult top::SqueezeOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int64_t i = 0; i < num_element; i++) {
    p.outputs[0][i] = p.inputs[0][i];
  }
  return success();
}

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


bool top::SiLUOp::isEltwise() {
  return true;
}

int64_t top::SiLUOp::getFLOPs() { return module::getNumElements(getOutput()) * 5; }

LogicalResult top::SiLUOp::init(InferenceParameter &p) { return success(); }
void top::SiLUOp::deinit(InferenceParameter &p) {}

LogicalResult top::SiLUOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = val / (1 + std::exp(-val));
  }
  return success();
}

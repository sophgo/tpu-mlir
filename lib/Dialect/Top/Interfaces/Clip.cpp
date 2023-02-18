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


bool top::ClipOp::isEltwise() {
  return false;
}

int64_t top::ClipOp::getFLOPs() { return 0; }

LogicalResult top::ClipOp::init(InferenceParameter &p) { return success(); }
void top::ClipOp::deinit(InferenceParameter &p) {}

LogicalResult top::ClipOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getOutput());
  auto min_v = static_cast<float>(getMin().convertToDouble());
  auto max_v = static_cast<float>(getMax().convertToDouble());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::min(max_v, std::max(min_v, val));
  }
  return success();
}

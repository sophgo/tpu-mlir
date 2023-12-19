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
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace mlir;

LogicalResult tpu::ModOp::init(InferenceParameter &p) { return success(); }
void tpu::ModOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ModOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInputs()[0]);
  // auto num_element1 = module::getNumElements(getInputs()[0]);
  // auto num_output = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val0 = p.inputs[0][i];
    auto val1 = p.inputs[2][i];
    p.outputs[0][i] = std::fmod(val0, val1);
  }
  return success();
}

LogicalResult tpu::ModOp::LocalGenSupport() {
  if (!(module::isBM1684XFamily() || module::isSG2260Family())) {
    return failure();
  }
  return BroadCastBinaryLocalGenSupport(getOperation());
}

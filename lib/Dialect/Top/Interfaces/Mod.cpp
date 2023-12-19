//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::ModOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult top::ModOp::init(InferenceParameter &p) { return success(); }

void top::ModOp::deinit(InferenceParameter &p) {}

LogicalResult top::ModOp::inference(InferenceParameter &p) {
  if (getInputs().size() != 2) {
    return failure();
  }
  auto input_num0 = module::getNumElements(getInputs()[0]);
  auto input_num1 = module::getNumElements(getInputs()[1]);
  auto output_num = module::getNumElements(getOutput());
  assert(input_num0 == output_num);
  assert(input_num1 == output_num);

#pragma omp parallel for schedule(static, omp_schedule(output_num))
  for (int i = 0; i < output_num; ++i) {
    auto val1 = p.inputs[0][i];
    auto val2 = p.inputs[1][i];
    p.outputs[0][i] = std::fmod(val1, val2);
  }
  return success();
}

void top::ModOp::shape_inference() {
  broadcast_shape_inference(getOperation());
  for (int i = 0; i < getNumOperands(); i++) {
    auto value = getInputs()[i];
    broadcast_tensor_reshape(getOutput(), value);
  }
}

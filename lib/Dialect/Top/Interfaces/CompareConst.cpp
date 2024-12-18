//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::CompareConstOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::CompareConstOp::init(InferenceParameter &p) {
  return success();
}
void top::CompareConstOp::deinit(InferenceParameter &p) {}

LogicalResult top::CompareConstOp::inference(InferenceParameter &p) {
  auto output_shape = computer_broadcast_shape(getOperation());
  module::setShape(getOutput(), output_shape);
  broadcast_shape_inference(getOperation());
  const auto num_element = module::getNumElements(getOutput());
  const float const_val_ = getConstVal().convertToDouble();
  if (!getInversed()) {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
    for (int i = 0; i < num_element; ++i) {
      p.outputs[0][i] = compare(p.inputs[0][i], const_val_, getMode());
    }
  } else {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
    for (int i = 0; i < num_element; ++i) {
      p.outputs[0][i] = compare(const_val_, p.inputs[0][i], getMode());
    }
  }
  return success();
}

void top::CompareConstOp::shape_inference() {
  common_shape_inference(getOperation());
  if (module::isShape(getInput())) {
    auto input_shape_v = module::getShapeTensorValue(getInput());
    auto out_shape = module::getShape(getOutput());
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), {input_shape_v}, out_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}

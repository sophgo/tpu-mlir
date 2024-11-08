//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::AddConstOp::getFLOPs() {
  return module::getNumElements(getOutput()) * (1 + (getDoRelu() ? 1 : 0));
}

LogicalResult top::AddConstOp::init(InferenceParameter &p) { return success(); }
void top::AddConstOp::deinit(InferenceParameter &p) {}

LogicalResult top::AddConstOp::inference(InferenceParameter &p) {
  auto output_shape = computer_broadcast_shape(getOperation());
  module::setShape(getOutput(), output_shape);
  const int64_t num_elem = module::getNumElements(getOutput());
  const float const_val_ = getConstVal().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = p.inputs[0][i] + const_val_;
  }
  if (getDoRelu()) {
    auto limit = getReluLimit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0], num_elem, limit);
  }
  return success();
}

void top::AddConstOp::shape_inference() {
  common_shape_inference(getOperation());
  auto output_shape = module::getShape(getOutput());
  if (module::isShape(getInput())) {
    auto input_v = module::getShapeTensorValue(getInput());
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), {input_v}, output_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}

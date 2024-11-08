//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ClipOp::getFLOPs() {
  return 2 * module::getNumElements(getOutput());
}

LogicalResult top::ClipOp::init(InferenceParameter &p) { return success(); }
void top::ClipOp::deinit(InferenceParameter &p) {}

LogicalResult top::ClipOp::inference(InferenceParameter &p) {
  auto output_shape = computer_broadcast_shape(getOperation());
  module::setShape(getOutput(), output_shape);
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

void top::ClipOp::shape_inference() {
  common_shape_inference(getOperation());

  auto out_shape = module::getShape(getOutput());
  if (module::isShape(getInputs())) {
    auto input_v = module::getShapeTensorValue(getInputs());
    auto output_v =
        module::commonShapeValInfer(getOperation(), {input_v}, out_shape);
    module::bindShapeTensorValue(getOutput(), output_v);
  }
}

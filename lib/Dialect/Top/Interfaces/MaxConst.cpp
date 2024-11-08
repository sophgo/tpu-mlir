//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::MaxConstOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::MaxConstOp::init(InferenceParameter &p) { return success(); }
void top::MaxConstOp::deinit(InferenceParameter &p) {}

LogicalResult top::MaxConstOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  const int64_t num_elem = module::getNumElements(getOutput());
  const float const_val_ = getConstVal().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = std::max(p.inputs[0][i], const_val_);
  }
  return success();
}

void top::MaxConstOp::shape_inference() {
  common_shape_inference(getOperation());
  if (module::isShape(getInput())) {
    auto input_v = module::getShapeTensorValue(getInput());
    auto out_shape = module::getShape(getOutput());
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), {input_v}, out_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::SubConstOp::getFLOPs() {
  return module::getNumElements(getOutput()) * (1 + (getDoRelu() ? 1 : 0));
}

LogicalResult top::SubConstOp::init(InferenceParameter &p) { return success(); }
void top::SubConstOp::deinit(InferenceParameter &p) {}

LogicalResult top::SubConstOp::inference(InferenceParameter &p) {
  const int64_t num_elem = module::getNumElements(getOutput());
  const float const_val_ = getConstVal().convertToDouble();
  const bool is_reverse_ = getIsReverse();
  if (is_reverse_) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = const_val_ - p.inputs[0][i];
    }
  } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = p.inputs[0][i] - const_val_;
    }
  }
  if (getDoRelu()) {
    auto limit = getReluLimit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0], num_elem, limit);
  }
  return success();
}

void top::SubConstOp::shape_inference() {
  common_shape_inference(getOperation());
  if (module::isShape(getInput())) {
    std::vector<std::vector<int64_t>> input_shapes_v;
    auto input_shape_v = module::getShapeTensorValue(getInput());
    input_shapes_v.push_back(input_shape_v);
    auto out_shape = module::getShape(getOutput());
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), input_shapes_v, out_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}

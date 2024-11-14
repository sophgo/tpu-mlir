
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"

int64_t top::DivConstOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::DivConstOp::init(InferenceParameter &p) { return success(); }
void top::DivConstOp::deinit(InferenceParameter &p) {}

LogicalResult top::DivConstOp::inference(InferenceParameter &p) {
  const int64_t num_elem = module::getNumElements(getOutput());
  const float const_val_ = getConstVal().convertToDouble();
  const bool is_reverse_ = getIsReverse();
  if (is_reverse_) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = std::floor(const_val_ / p.inputs[0][i]);
    }
  } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = std::floor(p.inputs[0][i] / const_val_);
    }
  }
  if (getDoRelu()) {
    auto limit = getReluLimit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0], num_elem, limit);
  }
  return success();
}

void top::DivConstOp::shape_inference() {
  common_shape_inference(getOperation());
  if (module::isShape(getInput())) {
    auto input_v = module::getShapeTensorValue(getInput());
    auto out_shape = module::getShape(getOutput());
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), {input_v}, out_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}

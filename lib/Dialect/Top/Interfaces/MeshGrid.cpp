//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::MeshGridOp::getFLOPs() { return 0; }

LogicalResult top::MeshGridOp::init(InferenceParameter &p) { return success(); }
void top::MeshGridOp::deinit(InferenceParameter &p) {}

LogicalResult top::MeshGridOp::inference(InferenceParameter &p) {
  auto shape = module::getShape(getOutputs()[0]);
  int64_t num = getInputs().size();
  auto num_element = module::getNumElements(getOutputs()[0]);
  int64_t outer = 1;
  for (int j = 0; j < num; ++j) {
    int64_t inner = num_element / outer / shape[j];
    int in_j = getIsReverse() ? num - 1 - j : j;
#pragma omp parallel for schedule(static, omp_schedule(outer *shape[j]))
    for (int i = 0; i < outer; ++i) {
      float *offset = p.outputs[in_j] + i * inner * shape[j];
      for (int k = 0; k < shape[j]; ++k) {
        float value = p.inputs[in_j][k];
        for (int m = 0; m < inner; ++m) {
          (offset + k * inner)[m] = value;
        }
      }
    }
    outer *= shape[j];
  }
  return success();
}

void top::MeshGridOp::shape_inference() {
  int64_t input_num = getInputs().size();
  int64_t length = 1;
  std::vector<int64_t> out_shape;
  for (int64_t i = 0; i < input_num; ++i) {
    int64_t idx = getIsReverse() ? (input_num - 1 - i) : i;
    auto shape = module::getShape(getInputs()[idx]);
    out_shape.push_back(shape[0]);
    length *= shape[0];
  }
  for (int i = 0; i < input_num; ++i) {
    auto out = getResult(i);
    module::setShapeOrVerify(out, out_shape);
  }
}

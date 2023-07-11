//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::SwapDimInnerOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::SwapDimInnerOp::init(InferenceParameter &p) {
  return success();
}
void top::SwapDimInnerOp::deinit(InferenceParameter &p) {}

LogicalResult top::SwapDimInnerOp::inference(InferenceParameter &p) {
  float *input_data = p.inputs[0];
  float *output_data = p.outputs[0];
  float *buffer = nullptr;

  std::vector<int64_t> input_shape = module::getShape(this->getInput());
  auto offsets = module::getI64Array(this->getOffset());
  int axis_num = 0;
  for (size_t i = 0; i < offsets->size(); ++i) {
    if (offsets->at(i) != 0) {
      axis_num++;
    }
  }
  if (axis_num > 1) {
    int64_t num_elements = module::getNumElements(this->getInput());
    buffer = new float[num_elements];
  }

  float *outs[2] = {axis_num % 2 ? output_data : buffer,
                    axis_num % 2 ? buffer : output_data};
  for (int i = 0; i < axis_num; ++i) {
    swap_dim_data(i == 0 ? input_data : outs[(1 - i % 2)], outs[i % 2],
                  input_shape, *offsets);
  }

  if (axis_num > 1) {
    delete[] buffer;
  }

  return success();
}

void top::SwapDimInnerOp::shape_inference() {
  common_shape_inference(getOperation());
}

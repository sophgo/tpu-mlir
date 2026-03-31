//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::InsertOp::init(InferenceParameter &p) { return success(); }
void tpu::InsertOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::InsertOp::inference(InferenceParameter &p) {
  float *input_data = p.inputs[0];
  float *dst_data = p.inputs[1];
  float *output_data = p.outputs[0];

  std::vector<int64_t> input_shape = module::getShape(this->getInput());
  std::vector<int64_t> dst_shape = module::getShape(this->getRhs());
  int64_t axis = this->getAxis();
  int64_t offset = this->getOffset();

  if (axis < 0 || axis >= input_shape.size()) {
    return failure();
  }
  if (offset < 0 || offset >= input_shape[axis]) {
    return failure();
  }
  if (input_shape.size() != dst_shape.size()) {
    return failure();
  }
  for (int i = 0; i < input_shape.size(); i++) {
    if (i != axis && input_shape[i] != dst_shape[i]) {
      return failure();
    }
  }
  if (offset + dst_shape[axis] > input_shape[axis]) {
    return failure();
  }

  insert_replace(input_data, dst_data, output_data, input_shape, dst_shape,
                 axis, offset);

  return success();
}

bool tpu::InsertOp::support_multi_core() { return false; }

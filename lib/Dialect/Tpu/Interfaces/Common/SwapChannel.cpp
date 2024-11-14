//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::SwapChannelOp::init(InferenceParameter &p) {
  return success();
}
void tpu::SwapChannelOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::SwapChannelOp::inference(InferenceParameter &p) {
  float *input_data = p.inputs[0];
  float *output_data = p.outputs[0];
  auto order = module::getI64Array(this->getChannelOrder());
  auto input_shape = module::getShape(this->getInput());
  int64_t n = input_shape[0];
  int64_t c = input_shape[1];
  int64_t frame_size = input_shape[2] * input_shape[3];
  int batch_length = c * frame_size;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; j++) {
      float *p_in = input_data + i * batch_length + frame_size * order->at(j);
      float *p_out = output_data + i * batch_length + frame_size * j;
      memcpy((void *)p_out, (void *)p_in, frame_size * sizeof(float));
    }
  }
  return success();
}

LogicalResult tpu::SwapChannelOp::LocalGenSupport() {
  if (!module::isCV18xx()) {
    return failure();
  }
  // return success();
  return failure();
}

bool tpu::SwapChannelOp::support_multi_core() { return false; }

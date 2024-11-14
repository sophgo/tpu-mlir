//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::ShuffleChannelOp::init(InferenceParameter &p) {
  return success();
}
void tpu::ShuffleChannelOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ShuffleChannelOp::inference(InferenceParameter &p) {
  int64_t group = this->getGroup();
  // auto input = this->getInput();
  float *input_data = p.inputs[0];
  float *output_data = p.outputs[0];
  auto input_shape = module::getShape(this->getInput());
  int64_t n = input_shape[0];
  int64_t c = input_shape[1];
  int64_t frame_size = input_shape[2] * input_shape[3];

  int batch_length = frame_size * c;
  int group_column = int(c / group);
  if (c % group != 0) {
    llvm::errs() << "Error: Wrong group size, c=" << c << ", group =" << group;
    llvm_unreachable("wrong group");
  }

  for (int i = 0; i < n; ++i) {
    float *p_in = input_data + i * batch_length;
    float *p_out = output_data + i * batch_length;
    for (int j = 0; j < group; ++j) // 2
    {
      for (int k = 0; k < group_column; ++k) // 3
      {
        float *p_i = p_in + (j * group_column + k) * frame_size;
        float *p_o = p_out + (k * group + j) * frame_size;
        std::memcpy(p_o, p_i, frame_size * sizeof(float));
      }
    }
  }
  return success();
}

bool tpu::ShuffleChannelOp::support_multi_core() { return false; }

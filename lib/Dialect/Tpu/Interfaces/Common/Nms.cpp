//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"

LogicalResult tpu::NmsOp::init(InferenceParameter &p) { return success(); }
void tpu::NmsOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::NmsOp::inference(InferenceParameter &p) {
  NmsParam param;
  int input_size = getInputs().size();
  assert(input_size >= 2);
  if (input_size >= 3) {
    param.max_output_boxes_per_class = p.inputs[2][0];
  } else {
    param.max_output_boxes_per_class = getMaxOutputSize();
  }
  param.center_point_box = 0;
  std::vector<tensor_list_t> input_list(2);
  for (int i = 0; i < 2; ++i) {
    tensor_list_t input;
    input.ptr = p.inputs[i];
    input.size = module::getNumElements(getInputs()[i]);
    input.shape = module::getShape(getInputs()[i]);
    input_list[i] = input;
  }
  param.box = p.inputs[0];
  param.score = p.inputs[1];
  int output_size = module::getNumElements(getOutput());
  std::vector<float> output_tensor_data(output_size, 0);
  param.inputs = input_list;
  param.output = output_tensor_data.data();
  if (input_size >= 5) {
    param.iou_threshold = p.inputs[3][0];
    param.score_threshold = p.inputs[4][0];
  } else {
    param.iou_threshold = 0.5;
    param.score_threshold = 0.5;
  }
  NmsFunc func(param);
  auto true_num = func.invoke();
  auto tmp = (int *)output_tensor_data.data();
  for (int64_t j = 0; j < true_num; ++j) {
    p.outputs[0][j] = (float)tmp[j];
  }
  std::vector<int64_t> output_shape{0, 3};
  assert(true_num % 3 == 0);
  output_shape[0] = (true_num / 3);
  module::setShape(getOutput(), output_shape);
  return success();
}

bool tpu::NmsOp::support_multi_core() { return false; }

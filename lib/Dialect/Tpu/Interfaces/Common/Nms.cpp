//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"

LogicalResult tpu::NmsOp::init(InferenceParameter &p) {
  return success();
}

void tpu::NmsOp::deinit(InferenceParameter &p) {
}

LogicalResult tpu::NmsOp::inference(InferenceParameter &p) {
  NmsParam param;
  param.max_output_boxes_per_class = getMaxOutputSize();
  param.center_point_box = 0;
  int input_size = getInputs().size();
  std::vector<tensor_list_t> input_list(input_size);
  for (int i = 0; i < getInputs().size(); ++i) {
    tensor_list_t input;
    input.ptr = p.inputs[0];
    input.size = module::getNumElements(getInputs()[i]);
    input.shape = module::getShape(getInputs()[i]);
    input_list[i] = input;
  }
  param.box = p.inputs[0];
  param.score = p.inputs[1];
  int output_size = module::getNumElements(getOutput());
  ;
  float output_tensor_data[output_size] = {0};
  param.inputs = input_list;
  param.output = output_tensor_data;
  param.iou_threshold = p.inputs[3][0];
  param.score_threshold = p.inputs[4][0];
  NmsFunc func(param);
  auto true_num = func.invoke();
  auto tmp = (int *)output_tensor_data;
  for (int64_t j = 0; j < true_num; ++j) {
    p.outputs[0][j] = (float)tmp[j];
  }
  return success();
}

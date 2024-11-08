//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"
int64_t top::NmsOp::getFLOPs() { return 0; }

LogicalResult top::NmsOp::init(InferenceParameter &p) { return success(); }

void top::NmsOp::deinit(InferenceParameter &p) {}

LogicalResult top::NmsOp::inference(InferenceParameter &p) {
  NmsParam param;
  int input_size = getInputs().size();
  ASSERT_THIS(input_size >= 2);
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
  int *tmp = (int *)output_tensor_data.data();
  for (int64_t j = 0; j < true_num; ++j) {
    p.outputs[0][j] = (float)tmp[j];
  }
  std::vector<int64_t> output_shape{0, 3};
  assert(true_num % 3 == 0);
  output_shape[0] = (true_num / 3);
  module::setShape(getOutput(), output_shape);
  return success();
}

void top::NmsOp::shape_inference() {
  int input_size = getInputs().size();
  ASSERT_THIS(input_size >= 2);
  int num_batch = module::getShape(getInputs()[1])[0];
  int num_class = module::getShape(getInputs()[1])[1];
  int spatial_dimension = module::getShape(getInputs()[1])[2];
  int64_t max_output_size_per_class = 0;
  if (input_size >= 3 && module::isShape(getInputs()[2])) {
    auto vec = module::getShapeTensorValue(getInputs()[2]);
    ASSERT_THIS(vec.size() == 1);
    max_output_size_per_class = vec[0];
  } else {
    max_output_size_per_class = getMaxOutputSize();
  }
  // update max_output_size_per_class, such as 2**31 -1
  if (max_output_size_per_class > spatial_dimension)
    max_output_size_per_class = spatial_dimension;
  std::vector<int64_t> output_shape{0, 3};
  output_shape[0] = num_batch * num_class * max_output_size_per_class;
  module::setShapeOrVerify(getOutput(), output_shape);
  // set top run mode to dynamic
  module::setTopRunMode(module::TopRunMode::DYNAMIC);
}

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
  if(module::isWeight(getInputs()[2])){
    param.max_output_boxes_per_class = getMaxOutputSize() ;
  } else{
    param.max_output_boxes_per_class = (int64_t)p.inputs[2][0];
  }
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
std::vector<float> output_tensor_data(output_size, 0);
  param.inputs = input_list;
  param.output = output_tensor_data.data();
  param.iou_threshold = p.inputs[3][0];
  param.score_threshold = p.inputs[4][0];
  NmsFunc func(param);
  auto true_num = func.invoke();
  auto tmp = (int *)output_tensor_data.data();
  for (int64_t j = 0; j < true_num; ++j) {
    p.outputs[0][j] = (float)tmp[j];
  }
  return success();
}

void top::NmsOp::shape_inference() {
  int class_num = module::getShape(getInputs()[1])[1];
  int max_output_size_per_class = 0;
  if (module::isShape(getInputs()[2])) {
    auto vec = module::getShapeTensorValue(getInputs()[2]);
    assert(vec.size() == 1);
    max_output_size_per_class = vec[0];
  } else {
    max_output_size_per_class = getMaxOutputSize();
  }
  std::vector<int64_t> output_shape{0,3};
  output_shape[0] = class_num * max_output_size_per_class;
  module::setShapeOrVerify(getOutput(), output_shape);
}

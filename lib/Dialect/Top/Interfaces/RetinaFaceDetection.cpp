//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"

int64_t top::RetinaFaceDetectionOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::RetinaFaceDetectionOp::init(InferenceParameter &p) {
  return success();
}
void top::RetinaFaceDetectionOp::deinit(InferenceParameter &p) {}

LogicalResult top::RetinaFaceDetectionOp::inference(InferenceParameter &p) {
  RetinaFaceDetectionParam param;
  param.confidence_threshold = getConfidenceThreshold().convertToDouble();
  param.keep_topk = getKeepTopk();
  param.nms_threshold = getNmsThreshold().convertToDouble();
  RetinaFaceDetectionFunc func;
  std::vector<tensor_list_t> inputs;
  for (size_t i = 0; i < getInputs().size(); i++) {
    tensor_list_t tensor;
    tensor.ptr = p.inputs[i];
    tensor.size = module::getNumElements(getInputs()[i]);
    tensor.shape = module::getShape(getInputs()[i]);
    inputs.emplace_back(std::move(tensor));
  }
  tensor_list_t output;
  output.ptr = p.outputs[0];
  output.size = module::getNumElements(getOutput());
  output.shape = module::getShape(getOutput());
  func.setup(inputs, output, param);
  func.invoke();
  return success();
}

void top::RetinaFaceDetectionOp::shape_inference() {
  int64_t keep_topk = getKeepTopk();
  auto input_shape = module::getShape(getInputs()[0]);
  int64_t batch = input_shape[0];

  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(batch);
  out_shape.push_back(1);
  out_shape.push_back(keep_topk);
  out_shape.push_back(15);
  //(x1, y1, x2, y2, score,
  //     landmark_pred_x1, landmark_pred_y1, landmark_pred_x2, landmark_pred_y2,
  //     landmark_pred_x3, landmark_pred_y3,
  //      landmark_pred_x4, landmark_pred_y4, landmark_pred_x5,
  //      landmark_pred_y5)
  module::setShapeOrVerify(getOutput(), out_shape);
}

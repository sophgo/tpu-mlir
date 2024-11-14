//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"

int64_t top::FrcnDetectionOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::FrcnDetectionOp::init(InferenceParameter &p) {
  return success();
}
void top::FrcnDetectionOp::deinit(InferenceParameter &p) {}

LogicalResult top::FrcnDetectionOp::inference(InferenceParameter &p) {
  FrcnDetParam param;
  param.class_num = getClassNum();
  param.keep_topk = getKeepTopk();
  param.nms_threshold = getNmsThreshold().convertToDouble();
  param.obj_threshold = getObjThreshold().convertToDouble();
  for (size_t i = 0; i < getInputs().size(); ++i) {
    tensor_list_t tensor_list;
    tensor_list.ptr = p.inputs[i];
    tensor_list.size = module::getNumElements(getInputs()[i]);
    tensor_list.shape = module::getShape(getInputs()[i]);
    param.inputs.emplace_back(std::move(tensor_list));
  }
  param.output.ptr = p.outputs[0];
  param.output.size = module::getNumElements(getOutput());
  param.output.shape = module::getShape(getOutput());
  FrcnDetctionFunc func(param);
  func.invoke();
  return success();
}

void top::FrcnDetectionOp::shape_inference() {
  auto input_shape = module::getShape(getInputs()[2]);
  int64_t batch = input_shape[0];
  int64_t keep_topk = getKeepTopk();

  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(batch);
  out_shape.push_back(1);
  out_shape.push_back(keep_topk);
  out_shape.push_back(6);
  //(bbox.x1, bbox.y1, bbox.x2, bbox.y2, class_num, score)
  module::setShapeOrVerify(getOutput(), out_shape);
}

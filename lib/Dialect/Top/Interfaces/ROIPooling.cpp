//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"

int64_t top::ROIPoolingOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::ROIPoolingOp::init(InferenceParameter &p) {
  return success();
}
void top::ROIPoolingOp::deinit(InferenceParameter &p) {}

LogicalResult top::ROIPoolingOp::inference(InferenceParameter &p) {
  ROIPoolingParam param;
  param.pooled_h = getPooledH();
  param.pooled_w = getPooledW();
  param.spatial_scale = getSpatialScale().convertToDouble();
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
  ROIPoolingFunc func(param);
  func.invoke();
  return success();
}

void top::ROIPoolingOp::shape_inference() {
  int64_t pooled_h = getPooledH();
  int64_t pooled_w = getPooledW();
  auto input_shape = module::getShape(getInputs()[0]);
  auto roi_shape = module::getShape(getInputs()[1]);
  int64_t channel = input_shape[1];
  int64_t num_rois = roi_shape[2];

  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(num_rois);
  out_shape.push_back(channel);
  out_shape.push_back(pooled_h);
  out_shape.push_back(pooled_w);
  module::setShapeOrVerify(getOutput(), out_shape);
}

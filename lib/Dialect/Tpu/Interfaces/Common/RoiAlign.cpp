//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"

LogicalResult tpu::RoiAlignOp::init(InferenceParameter &p) { return success(); }

void tpu::RoiAlignOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RoiAlignOp::inference(InferenceParameter &p) {
  auto batch = module::getShape(getRois())[0];
  auto shape = module::getShape(getInput()).vec();
  uint32_t ndims = shape.size();
  shape[0] = batch;
  shape[ndims - 1] = getOutputWidth();
  shape[ndims - 2] = getOutputHeight();
  module::setShape(getOutput(), shape);
  RoiAlignParam param;
  param.mode = getMode().str() == "Avg" ? RoiAlignAvgMode : RoiAlignMaxMode;
  param.pooled_h = getOutputHeight();
  param.pooled_w = getOutputWidth();
  param.sampling_ratio = getSamplingRatio();
  param.spatial_scale = getSpatialScale().convertToDouble();
  param.aligned = getAlignCorners();
  tensor_list_t tensor_list0, tensor_list1;
  tensor_list0.ptr = p.inputs[0];
  tensor_list0.size = module::getNumElements(getInput());
  tensor_list0.shape = module::getShape(getInput());
  param.inputs.emplace_back(std::move(tensor_list0));
  tensor_list1.ptr = p.inputs[1];
  tensor_list1.size = module::getNumElements(getRois());
  tensor_list1.shape = module::getShape(getRois());
  param.inputs.emplace_back(std::move(tensor_list1));
  param.output.ptr = p.outputs[0];
  param.output.size = module::getNumElements(getOutput());
  param.output.shape = module::getShape(getOutput());
  RoiAlignFunc func(param);
  func.invoke();
  return success();
}

bool tpu::RoiAlignOp::support_multi_core() { return false; }

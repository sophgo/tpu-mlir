//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
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
    module::getShapeVec(getInputs()[i], tensor_list.shape);
    param.inputs.emplace_back(std::move(tensor_list));
  }
  param.output.ptr = p.outputs[0];
  param.output.size = module::getNumElements(getOutput());
  module::getShapeVec(getOutput(), param.output.shape);
  ROIPoolingFunc func(param);
  func.invoke();
  return success();
}

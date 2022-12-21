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
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/GenericCpuFunc.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

int64_t top::ROIPoolingOp::getFLOPs() {
  return Module::getNumElements(output());
}

LogicalResult top::ROIPoolingOp::init(InferenceParameter &p) {
  return success();
}
void top::ROIPoolingOp::deinit(InferenceParameter &p) {}

LogicalResult top::ROIPoolingOp::inference(InferenceParameter &p) {
  ROIPoolingParam param;
  param.pooled_h = pooled_h();
  param.pooled_w = pooled_w();
  param.spatial_scale = spatial_scale().convertToDouble();
  for (size_t i = 0; i < inputs().size(); ++i) {
    tensor_list_t tensor_list;
    tensor_list.ptr = p.inputs[i];
    tensor_list.size = Module::getNumElements(inputs()[i]);
    Module::getShapeVec(inputs()[i], tensor_list.shape);
    param.inputs.emplace_back(std::move(tensor_list));
  }
  param.output.ptr = p.outputs[0];
  param.output.size = Module::getNumElements(output());
  Module::getShapeVec(output(), param.output.shape);
  ROIPoolingFunc func(param);
  func.invoke();
  return success();
}

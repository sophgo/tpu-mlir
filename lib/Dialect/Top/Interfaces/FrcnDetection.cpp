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



int64_t top::FrcnDetectionOp::getFLOPs() {
  return module::getNumElements(output());
}

LogicalResult top::FrcnDetectionOp::init(InferenceParameter &p) {
  return success();
}
void top::FrcnDetectionOp::deinit(InferenceParameter &p) {}

LogicalResult top::FrcnDetectionOp::inference(InferenceParameter &p) {
  FrcnDetParam param;
  param.class_num = class_num();
  param.keep_topk = keep_topk();
  param.nms_threshold = nms_threshold().convertToDouble();
  param.obj_threshold = obj_threshold().convertToDouble();
  for (size_t i = 0; i < inputs().size(); ++i) {
    tensor_list_t tensor_list;
    tensor_list.ptr = p.inputs[i];
    tensor_list.size = module::getNumElements(inputs()[i]);
    module::getShapeVec(inputs()[i], tensor_list.shape);
    param.inputs.emplace_back(std::move(tensor_list));
  }
  param.output.ptr = p.outputs[0];
  param.output.size = module::getNumElements(output());
  module::getShapeVec(output(), param.output.shape);
  FrcnDetctionFunc func(param);
  func.invoke();
  return success();
}

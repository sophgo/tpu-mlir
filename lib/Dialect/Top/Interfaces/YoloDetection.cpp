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

int64_t top::YoloDetectionOp::getFLOPs() {
  return Module::getNumElements(output());
}

LogicalResult top::YoloDetectionOp::init(InferenceParameter &p) {
  return success();
}
void top::YoloDetectionOp::deinit(InferenceParameter &p) {}

LogicalResult top::YoloDetectionOp::inference(InferenceParameter &p) {
  YoloDetParam param;
  param.class_num = class_num();
  param.net_input_h = net_input_h();
  param.net_input_w = net_input_w();
  param.keep_topk = keep_topk();
  param.nms_threshold = nms_threshold().convertToDouble();
  param.obj_threshold = obj_threshold().convertToDouble();
  param.tiny = tiny();
  param.yolo_v4 = yolo_v4();
  param.spp_net = spp_net();
  param.anchors = anchors().str();
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
  YoloDetectionFunc yolo_func(param);
  yolo_func.invoke();
  return success();
}

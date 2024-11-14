//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include <queue>
#include <vector>

LogicalResult tpu::YoloDetectionOp::init(InferenceParameter &p) {
  return success();
}
void tpu::YoloDetectionOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::YoloDetectionOp::inference(InferenceParameter &p) {
  YoloDetParam param;
  param.class_num = getClassNum();
  param.net_input_h = getNetInputH();
  param.net_input_w = getNetInputW();
  param.keep_topk = getKeepTopk();
  param.nms_threshold = getNmsThreshold().convertToDouble();
  param.obj_threshold = getObjThreshold().convertToDouble();
  param.anchors = *module::getF64Array(getAnchors());
  param.num_boxes = getNumBoxes();
  param.agnostic_nms = getAgnosticNms();
  auto inputs = getInputs();
  auto num_input = inputs.size();
  param.version = getVersion().str();
  for (int i = 0; i < param.num_boxes * num_input; i++) {
    param.mask.push_back(i);
  }
  for (size_t i = 0; i < num_input; ++i) {
    tensor_list_t tensor_list;
    tensor_list.ptr = p.inputs[i];
    tensor_list.size = module::getNumElements(inputs[i]);
    tensor_list.shape = module::getShape(inputs[i]);
    param.inputs.emplace_back(std::move(tensor_list));
  }
  param.output.ptr = p.outputs[0];
  param.output.size = module::getNumElements(getOutput());
  param.output.shape = module::getShape(getOutput());

  // empty process means the yolo layer comes from origin model but not
  // add_postprocess
  auto process = module::getPostprocess();
  if (process.empty()) {
    YoloDetectionFunc yolo_func(param);
    yolo_func.invoke();
  } else if ((process.starts_with("yolov5") || process.starts_with("yolov7")) &&
             param.inputs.size() == 1 && param.inputs[0].shape.size() == 3) {
    Yolov5DetectionFunc yolo_func(param);
    yolo_func.invoke();
  } else if (process.starts_with("yolov8") || process.starts_with("yolov11")) {
    Yolov8DetectionFunc yolo_func(param);
    yolo_func.invoke();
  } else {
    YoloDetectionFunc_v2 yolo_func(param);
    yolo_func.invoke();
  }
  return success();
}

bool tpu::YoloDetectionOp::support_multi_core() { return false; }

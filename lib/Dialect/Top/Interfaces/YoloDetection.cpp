//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"

int64_t top::YoloDetectionOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::YoloDetectionOp::init(InferenceParameter &p) {
  return success();
}
void top::YoloDetectionOp::deinit(InferenceParameter &p) {}

LogicalResult top::YoloDetectionOp::inference(InferenceParameter &p) {
  auto num_input = getInputs().size();
  YoloDetParam param;
  param.class_num = getClassNum();
  param.net_input_h = getNetInputH();
  param.net_input_w = getNetInputW();
  param.keep_topk = getKeepTopk();
  param.nms_threshold = getNmsThreshold().convertToDouble();
  param.obj_threshold = getObjThreshold().convertToDouble();
  param.agnostic_nms = getAgnosticNms();
  param.anchors = *module::getF64Array(getAnchors());
  param.num_boxes = getNumBoxes();
  param.version = getVersion().str();
  for (int i = 0; i < param.num_boxes * num_input; i++) {
    param.mask.push_back(i);
  }
  for (size_t i = 0; i < num_input; ++i) {
    tensor_list_t tensor_list;
    tensor_list.ptr = p.inputs[i];
    tensor_list.size = module::getNumElements(getInputs()[i]);
    tensor_list.shape = module::getShape(getInputs()[i]);
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
  } else if (process.starts_with("yolov5") && p.inputs.size() == 1 &&
             param.inputs[0].shape.size() == 3) {
    Yolov5DetectionFunc yolo_func(param);
    yolo_func.invoke();
  } else if (process.starts_with("yolov8")) {
    Yolov8DetectionFunc yolo_func(param);
    yolo_func.invoke();
  } else {
    auto output_shape = module::getShape(this->getOutput());
    int64_t dim = output_shape.size();
    if (output_shape[dim - 1] == 6) {
      //(x, y, w, h, cls, score)
      YoloDetectionFunc yolo_func(param);
      yolo_func.invoke();
    } else {
      //(batch_idx, cls, score, x, y, w, h)
      YoloDetectionFunc_v2 yolo_func(param);
      yolo_func.invoke();
    }
  }
  return success();
}

void top::YoloDetectionOp::shape_inference() {
  auto in0_shape = module::getShape(getInputs()[0]);
  int64_t batch = in0_shape[0];
  int64_t keep_topk = getKeepTopk();
  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(batch);
  out_shape.push_back(1);
  out_shape.push_back(keep_topk);
  if (batch == 1) {
    out_shape.push_back(6);
    //(x, y, w, h, cls, score)
  } else {
    out_shape.push_back(7);
    // (batch_idx, cls, score, x, y, w, h)
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/GenericCpuFunc.h"
#include <algorithm>
#include <queue>
#include <vector>

LogicalResult tpu::YoloDetectionOp::init(InferenceParameter &p) { return success(); }
void tpu::YoloDetectionOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::YoloDetectionOp::inference(InferenceParameter &p) {
  YoloDetParam param;
  param.class_num = getClassNum();
  param.net_input_h = getNetInputH();
  param.net_input_w = getNetInputW();
  param.keep_topk = getKeepTopk();
  param.nms_threshold = getNmsThreshold().convertToDouble();
  param.obj_threshold = getObjThreshold().convertToDouble();
  param.tiny = getTiny();
  param.yolo_v4 = getYoloV4();
  param.spp_net = getSppNet();
  param.anchors = getAnchors().str();
  param.num_boxes = getNumBoxes();
  param.mask_group_size = getMaskGroupSize();
  ArrayAttr&& mask = getMask();
  for (uint32_t i = 0; i < mask.size(); i++)
    param.mask[i] = (float)(mask[i].cast<IntegerAttr>().getInt());
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
  if (getFlag()) {
    int total_num = 0;
    Yolo_v2_DetectionFunc yolo_v2_func(param);
    yolo_v2_func.invoke(total_num);
  } else {
    YoloDetectionFunc yolo_func(param);
    yolo_func.invoke();
  }
  return success();
}

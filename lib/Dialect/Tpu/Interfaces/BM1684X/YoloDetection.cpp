//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif
typedef struct yolov3_detect_out_spec {
  int input_num;
  int batch_num;
  int num_classes;
  int num_boxes;
  int mask_group_size;
  int keep_top_k;
  float nms_threshold;
  float confidence_threshold;
  float bias[18];
  float anchor_scale[3];
  float mask[9];
  int yolo_box_flag; // 0: yolov3_detect_out, 1:paddle_yolo_box
  int clip_bbox;     // used for paddle yolo_box 1:true, 0:false
  float scale;       // used for paddle yolo_box
  int clip_im_height;
  int clip_im_width;
} yolov3_detect_out_spec_t;

#ifdef __cplusplus
}
#endif
// =========================================
// GlobalGenInterface
// =========================================
void tpu::YoloDetectionOp::codegen_global_bm1684x() {
  llvm_unreachable("Only support dynamic codegen");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::YoloDetectionOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(yolov3_detect_out_spec_t);
  yolov3_detect_out_spec_t spec = {0};
  spec.input_num = getInputs().size();
  spec.batch_num = module::getShape(getInputs()[0])[0];
  spec.num_classes = getClassNum();
  spec.num_boxes = getNumBoxes();
  spec.mask_group_size = spec.num_boxes;
  spec.keep_top_k = getKeepTopk();
  spec.nms_threshold = getNmsThreshold().convertToDouble();
  spec.confidence_threshold = getObjThreshold().convertToDouble();
  auto scale = module::getI64Array(getScale(), spec.num_boxes, 0);
  auto mask =
      module::getI64Array(getMask(), spec.num_boxes * spec.input_num, 0);
  auto anchors = module::getI64Array(getAnchors());
  for (uint32_t i = 0; i < scale->size(); i++) {
    spec.anchor_scale[i] = (float)(scale->at(i));
  }
  for (uint32_t i = 0; i < mask->size(); i++) {
    spec.mask[i] = (float)(mask->at(i));
  }
  for (uint32_t i = 0; i < anchors->size(); i++) {
    spec.bias[i] = (float)(anchors->at(i));
  }
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::YoloDetectionOp::get_fw_type_bm1684x() {
  return FW_BMNET_YOLOV3_DETECT_OUT;
}

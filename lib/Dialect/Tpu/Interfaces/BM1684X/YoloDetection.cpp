//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

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

typedef struct yolov3_detect_out_dyn_param {
  yolov3_detect_out_spec_t spec;
  unsigned long long buffer_addr;
  int detected_box_num;
} yolov3_detect_out_dyn_param_t;

typedef struct yolov5_detect_out_spec {
  int keep_top_k;
  float nms_threshold;
  float confidence_threshold;
  int agnostic_nms;
  int max_hw;
} yolov5_detect_out_spec_t;

typedef struct yolov5_decode_detect_out_spec {
  int input_num;
  int batch_num;
  int num_classes;
  int num_boxes;
  int keep_top_k;
  float nms_threshold;
  float confidence_threshold;
  float anchors[2 * MAX_YOLO_INPUT_NUM * MAX_YOLO_ANCHOR_NUM];
  float anchor_scale[MAX_YOLO_ANCHOR_NUM];
  int agnostic_nms;
} yolov5_decode_detect_out_spec_t;

typedef struct yolov8_detect_out_spec {
  int keep_top_k;
  float nms_threshold;
  float confidence_threshold;
  int agnostic_nms;
  int max_hw;
} yolov8_detect_out_spec_t;

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
  auto process = module::getPostprocess();
  if (process.starts_with("yolov5") &&
      module::getShape(getInputs()[0]).size() == 4) {
    if (!buffer)
      return sizeof(yolov5_decode_detect_out_spec_t);
    yolov5_decode_detect_out_spec_t spec = {0};
    spec.input_num = getInputs().size();
    spec.batch_num = module::getShape(getInputs()[0])[0];
    spec.num_classes = getClassNum();
    spec.num_boxes = getNumBoxes();
    spec.keep_top_k = getKeepTopk();
    spec.nms_threshold = getNmsThreshold().convertToDouble();
    spec.confidence_threshold = getObjThreshold().convertToDouble();
    spec.agnostic_nms = (int)getAgnosticNms();
    auto anchors = module::getF64Array(getAnchors());
    for (uint32_t i = 0; i < anchors->size(); i++) {
      spec.anchors[i] = (float)(anchors->at(i));
    }
    double width = (double)getNetInputW();
    for (uint32_t i = 0; i < spec.input_num; i++) {
      auto s = module::getShape(getInputs()[i]);
      assert(s.size() == 4);
      spec.anchor_scale[i] = (float)(width / s[3]);
    }
    return BM168x::dynamic_spec_to_buffer(buffer, spec);
  } else if (process.starts_with("yolov8") && getInputs().size() == 1 &&
             module::getShape(getInputs()[0]).size() == 3) {
    if (!buffer)
      return sizeof(yolov8_detect_out_spec_t);
    yolov8_detect_out_spec_t spec = {0};
    spec.keep_top_k = getKeepTopk();
    spec.nms_threshold = getNmsThreshold().convertToDouble();
    spec.confidence_threshold = getObjThreshold().convertToDouble();
    spec.agnostic_nms = (int)getAgnosticNms();
    spec.max_hw =
        !spec.agnostic_nms ? std::max(getNetInputW(), getNetInputH()) : 0;
    return BM168x::dynamic_spec_to_buffer(buffer, spec);

  } else if (process.starts_with("yolov5") && getInputs().size() == 1 &&
             module::getShape(getInputs()[0]).size() == 3) {
    if (!buffer)
      return sizeof(yolov5_detect_out_spec_t);
    yolov5_detect_out_spec_t spec = {0};
    spec.keep_top_k = getKeepTopk();
    spec.nms_threshold = getNmsThreshold().convertToDouble();
    spec.confidence_threshold = getObjThreshold().convertToDouble();
    spec.agnostic_nms = (int)getAgnosticNms();
    spec.max_hw =
        !spec.agnostic_nms ? std::max(getNetInputW(), getNetInputH()) : 0;
    return BM168x::dynamic_spec_to_buffer(buffer, spec);

  } else { // default
    if (!buffer)
      return sizeof(yolov3_detect_out_dyn_param_t);
    yolov3_detect_out_dyn_param_t param = {0};
    param.spec.input_num = getInputs().size();
    param.spec.batch_num = module::getShape(getInputs()[0])[0];
    param.spec.num_classes = getClassNum();
    param.spec.num_boxes = getNumBoxes();
    param.spec.mask_group_size = param.spec.num_boxes;
    param.spec.keep_top_k = getKeepTopk();
    param.spec.nms_threshold = getNmsThreshold().convertToDouble();
    param.spec.confidence_threshold = getObjThreshold().convertToDouble();
    auto anchors = module::getF64Array(getAnchors());
    double width = (double)getNetInputW();
    for (uint32_t i = 0; i < param.spec.input_num; i++) {
      auto s = module::getShape(getInputs()[i]);
      assert(s.size() == 4);
      param.spec.anchor_scale[i] = (float)(width / s[3]);
    }
    for (uint32_t i = 0; i < param.spec.input_num * param.spec.num_boxes; i++) {
      param.spec.mask[i] = (float)(i);
    }
    for (uint32_t i = 0; i < anchors->size(); i++) {
      param.spec.bias[i] = (float)(anchors->at(i));
    }
    param.buffer_addr = (module::isBM1688() || module::isSG2380())
                            ? module::getAddress(getBuffer())
                            : 0;
    return BM168x::dynamic_spec_to_buffer(buffer, param);
  }
}

int64_t tpu::YoloDetectionOp::get_fw_type_bm1684x() {
  auto process = module::getPostprocess();
  if (process.starts_with("yolov5")) {
    return FW_BMNET_YOLOV5_DETECT_OUT;
  } else if (process.starts_with("yolov8")) {
    return FW_BMNET_YOLOV8_DETECT_OUT;
  } else { // default
    return FW_BMNET_YOLOV3_DETECT_OUT;
  }
}

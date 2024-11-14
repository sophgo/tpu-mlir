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
typedef struct ssd_detect_out_spec {
  int num_classes;
  int share_location;
  int background_label_id;
  int code_type;
  int variance_encoded_in_target;
  int keep_top_k;
  float confidence_threshold;
  float nms_threshold;
  float eta;
  int top_k;
  int onnx_nms; // 1: onnx_nms
} ssd_detect_out_spec_t;

typedef struct ssd_detect_out_dyn_param {
  ssd_detect_out_spec_t spec;
  unsigned long long buffer_addr;
  int detected_box_num;
} ssd_detect_out_dyn_param_t;
#ifdef __cplusplus
}
#endif
// =========================================
// GlobalGenInterface
// =========================================
void tpu::DetectionOutputOp::codegen_global_bm1684x() {
  llvm_unreachable("Only support dynamic codegen");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::DetectionOutputOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(ssd_detect_out_dyn_param_t);
  ssd_detect_out_dyn_param_t param = {0};
  param.spec.num_classes = getNumClasses();
  param.spec.share_location = getShareLocation();
  param.spec.background_label_id = getBackgroundLabelId();
  std::string str_code_type = this->getCodeType().str();
  if (str_code_type == "CORNER") {
    param.spec.code_type = 1;
  } else if (str_code_type == "CENTER_SIZE") {
    param.spec.code_type = 2;
  } else if (str_code_type == "CORNER_SIZE") {
    param.spec.code_type = 3;
  } else {
    llvm_unreachable("code type wrong");
  }
  param.spec.variance_encoded_in_target =
      getVarianceEncodedInTarget().convertToDouble();
  param.spec.keep_top_k = getKeepTopK();
  param.spec.confidence_threshold = getConfidenceThreshold().convertToDouble();
  param.spec.nms_threshold = getNmsThreshold().convertToDouble();
  param.spec.eta = getEta().convertToDouble();
  param.spec.top_k = getTopK();
  param.spec.onnx_nms = getOnnxNms();
  param.buffer_addr = module::isBM1688() ? module::getAddress(getBuffer()) : 0;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::DetectionOutputOp::get_fw_type_bm1684x() {
  return FW_BMNET_SSD_DETECT_OUT;
}

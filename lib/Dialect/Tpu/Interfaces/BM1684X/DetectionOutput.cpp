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
typedef struct ssd_detect_out_spec {
  int   num_classes;
  int   share_location;
  int   background_label_id;
  int   code_type;
  int   variance_encoded_in_target;
  int   keep_top_k;
  float confidence_threshold;
  float nms_threshold;
  float eta;
  int   top_k;
  int  onnx_nms;// 1: onnx_nms
} ssd_detect_out_spec_t;

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
  if (!buffer) return sizeof(ssd_detect_out_spec_t);
  ssd_detect_out_spec_t spec = {0};
  spec.num_classes = getNumClasses();
  spec.share_location = getShareLocation();
  spec.background_label_id = getBackgroundLabelId();
  std::string str_code_type = this->getCodeType().str();
  if (str_code_type == "CORNER") {
    spec.code_type = 1;
  } else if (str_code_type == "CENTER_SIZE") {
    spec.code_type = 2;
  } else if (str_code_type == "CORNER_SIZE") {
    spec.code_type = 3;
  } else {
    llvm_unreachable("code type wrong");
  }
  spec.variance_encoded_in_target = getVarianceEncodedInTarget().convertToDouble();
  spec.keep_top_k = getKeepTopK();
  spec.confidence_threshold = getConfidenceThreshold().convertToDouble();
  spec.nms_threshold = getNmsThreshold().convertToDouble();
  spec.eta = getEta().convertToDouble();
  spec.top_k = getTopK();
  spec.onnx_nms = getOnnxNms();
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::DetectionOutputOp::get_fw_type_bm1684x() {
  return FW_BMNET_SSD_DETECT_OUT;
}

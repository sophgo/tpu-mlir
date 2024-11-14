//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::MaskRCNNGetBboxBOp::codegen_global_bm1684x() {
  llvm_unreachable("Only support dynamic codegen,as Superior_MaskRCNN involved "
                   "with sort/onnx_nms/roi_align");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::MaskRCNNGetBboxBOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) {
    return sizeof(MaskRCNN_get_bbox_B_global_param_t);
  }
  MaskRCNN_get_bbox_B_global_param_t param = {0};
  param.global_buffer_0_means = module::getAddress(getMeans());
  param.global_buffer_1_stds = module::getAddress(getStds());
  param.global_buffer_2_res_bbox = module::getAddress(getResBbox());
  param.global_buffer_3_res_bbox1 = module::getAddress(getResBbox1());
  param.global_buffer_4_res_bbox0 = module::getAddress(getResBbox0());
  param.global_buffer_5_res_score0 = module::getAddress(getResScore0());
  param.global_buffer_6_res_score1 = module::getAddress(getResScore1());
  param.global_buffer_7_res_score2 = module::getAddress(getResScore2());
  param.global_buffer_8_res_score3 = module::getAddress(getResScore3());
  param.global_buffer_9_res_label2 = module::getAddress(getResLabel2());
  param.global_buffer_10_result_list = module::getAddress(getResultList());
  param.global_buffer_11_keep_3nch = module::getAddress(getKeep_3nch());
  param.global_buffer_12_keep_u32_1h = module::getAddress(getKeepU32_1h());
  param.global_buffer_13_glb_buffer_boxes =
      module::getAddress(getGlbBufferBoxes());
  param.global_buffer_14_glb_buffer_scores =
      module::getAddress(getGlbBufferScores());
  param.global_buffer_15_glb_buffer_nms = module::getAddress(getGlbBufferNms());
  param.global_buffer_16_glb_buffer_nonzero =
      module::getAddress(getGlbBufferNonzero());
  param.global_buffer_17_result_valid_ind =
      module::getAddress(getResultValidInd());
  param.global_buffer_18_glb_lables = module::getAddress(getGlbLables());
  param.global_buffer_19_glb_lables_expand =
      module::getAddress(getGlbLablesExpand());
  param.spec.threshold_score_eq = getThresholdScoreEq().convertToDouble();
  param.spec.wh_ratio_log = getWhRatioLog().convertToDouble();
  param.spec.nms_iou_thr = getNmsIouThr().convertToDouble();
  param.spec.delta2bbox_means = getDelta2bboxMeans().convertToDouble();
  param.spec.delta2bbox_stds_0 = getDelta2bboxStds_0().convertToDouble();
  param.spec.delta2bbox_stds_1 = getDelta2bboxStds_1().convertToDouble();
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::MaskRCNNGetBboxBOp::get_fw_type_bm1684x() {
  return FW_BMNET_MASKRCNNGETBBOXB;
}

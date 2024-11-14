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
void tpu::MaskRCNNRPNGetBboxesOp::codegen_global_bm1684x() {
  llvm_unreachable("Only support dynamic codegen,as Superior_MaskRCNN involved "
                   "with sort/onnx_nms/roi_align");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::MaskRCNNRPNGetBboxesOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) {
    return sizeof(MaskRCNN_RPN_get_bboxes_global_param_t);
  }
  MaskRCNN_RPN_get_bboxes_global_param_t param = {0};
  param.global_buffer_0_batch_mlvl_scores =
      module::getAddress(getBatchMlvlScores());
  param.global_buffer_1_batch_mlvl_anchors =
      module::getAddress(getBatchMlvlAnchors());
  param.global_buffer_2_batch_mlvl_rpn_bbox_pred =
      module::getAddress(getBatchMlvlRpnBboxPred());
  param.global_buffer_3_batch_mlvl_proposals =
      module::getAddress(getBatchMlvlProposals());
  param.global_buffer_4_batch_mlvl_ids = module::getAddress(getBatchMlvlIds());
  param.global_buffer_5_glb_buffer_tmp_scores_stretched =
      module::getAddress(getGlbBufferTmpScoresStretched());
  param.global_buffer_6_glb_buffer_ranked_scores =
      module::getAddress(getGlbBufferRankedScores());
  param.global_buffer_7_glb_buffer_rank_inds_int32 =
      module::getAddress(getGlbBufferRankIndsInt32());
  param.global_buffer_8_glb_buffer_rank_inds_u32 =
      module::getAddress(getGlbBufferRankIndsU32());
  param.global_buffer_9_glb_topk_inds = module::getAddress(getGlbTopkInds());
  param.global_buffer_10_glb_buffer_gather_1 =
      module::getAddress(getGlbBufferGather_1());
  param.global_buffer_11_glb_buffer_gather_2 =
      module::getAddress(getGlbBufferGather_2());
  param.global_buffer_12_glb_buffer_rpn_bbox_permuted =
      module::getAddress(getGlbBufferRpnBboxPermuted());
  param.global_buffer_13_glb_buffer_nonzero =
      module::getAddress(getGlbBufferNonzero());
  param.global_buffer_14_result_valid_ind =
      module::getAddress(getResultValidInd());
  param.global_buffer_15_glb_buffer_gather_boxes =
      module::getAddress(getGlbBufferGatherBoxes());
  param.global_buffer_16_glb_buffer_gather_scores =
      module::getAddress(getGlbBufferGatherScores());
  param.global_buffer_17_keep_3nch = module::getAddress(getKeep_3nch());
  param.global_buffer_18_keep_u32_1h = module::getAddress(getKeepU32_1h());
  param.global_buffer_19_glb_buffer_boxes =
      module::getAddress(getGlbBufferBoxes());
  param.global_buffer_20_glb_buffer_scores =
      module::getAddress(getGlbBufferScores());
  param.global_buffer_21_glb_buffer_nms = module::getAddress(getGlbBufferNms());
  param.global_buffer_22_gather_mlvl_proposals =
      module::getAddress(getGatherMlvlProposals());
  param.global_buffer_23_gather_mlvl_scores =
      module::getAddress(getGatherMlvlScores());
  param.global_buffer_24_gather_mlvl_ids =
      module::getAddress(getGatherMlvlIds());
  param.global_buffer_25_glb_buffer_result_list =
      module::getAddress(getGlbBufferResultList());
  param.spec.delta2bbox_mean_0 = getDelta2bboxMean_0().convertToDouble();
  param.spec.delta2bbox_mean_1 = getDelta2bboxMean_1().convertToDouble();
  param.spec.delta2bbox_mean_2 = getDelta2bboxMean_2().convertToDouble();
  param.spec.delta2bbox_mean_3 = getDelta2bboxMean_3().convertToDouble();
  param.spec.delta2bbox_std_0 = getDelta2bboxStd_0().convertToDouble();
  param.spec.delta2bbox_std_1 = getDelta2bboxStd_1().convertToDouble();
  param.spec.delta2bbox_std_2 = getDelta2bboxStd_2().convertToDouble();
  param.spec.delta2bbox_std_3 = getDelta2bboxStd_3().convertToDouble();
  param.spec.delta2bbox_max_scalar_c =
      getDelta2bboxMaxScalarC().convertToDouble();
  param.spec.iou_threshold = getIouThreshold().convertToDouble();
  param.spec.conf_threshold = getConfThreshold().convertToDouble();
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::MaskRCNNRPNGetBboxesOp::get_fw_type_bm1684x() {
  return FW_BMNET_MASKRCNNRPNGETBBOXES;
}

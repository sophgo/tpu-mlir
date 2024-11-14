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
void tpu::MaskRCNNMaskPoolerOp::codegen_global_bm1684x() {
  llvm_unreachable("Only support dynamic codegen,as Superior_MaskRCNN involved "
                   "with sort/onnx_nms/roi_align");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::MaskRCNNMaskPoolerOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) {
    return sizeof(MaskRCNN_mask_pooler_global_param_t);
  }
  MaskRCNN_mask_pooler_global_param_t param = {0};
  param.global_buffer_0_ptr_rois_buff = module::getAddress(getPtrRoisBuff());
  param.global_buffer_1_result_filled_det_bboxes =
      module::getAddress(getResultFilledDetBboxes());
  param.global_buffer_2_result_filled_det_labels =
      module::getAddress(getResultFilledDetLabels());
  param.global_buffer_3_ptr_tmp_res = module::getAddress(getPtrTmpRes());
  param.global_buffer_4_ptr_rois_tmp = module::getAddress(getPtrRoisTmp());
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::MaskRCNNMaskPoolerOp::get_fw_type_bm1684x() {
  return FW_BMNET_MASKRCNNMASKPOOLER;
}

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
void tpu::MaskRCNNBboxPoolerOp::codegen_global_bm1684x() {
  llvm_unreachable("Only support dynamic codegen,as Superior_MaskRCNN involved "
                   "with sort/onnx_nms/roi_align");
}

// ===============1=======================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::MaskRCNNBboxPoolerOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) {
    return sizeof(MaskRCNN_bbox_pooler_global_param_t);
  }
  MaskRCNN_bbox_pooler_global_param_t param = {0};
  param.global_buffer_0_ptr_tmp_res = module::getAddress(getPtrTmpRes());
  param.global_buffer_1_ptr_rois_tmp = module::getAddress(getPtrRoisTmp());
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::MaskRCNNBboxPoolerOp::get_fw_type_bm1684x() {
  return FW_BMNET_MASKRCNNBBOXPOOLER;
}

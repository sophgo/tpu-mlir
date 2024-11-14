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

// =========================================
// GlobalGenInterface
// =========================================
void tpu::RoiAlignOp::codegen_global_bm1684x() {
  llvm_unreachable("Only support dynamic codegen");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::RoiAlignOp::dyn_codegen_global_bm1684x(void *buffer) {
  assert(getMode() != "Max");
  if (!buffer) {
    return sizeof(roi_align_spec_t);
  }
  roi_align_spec_t spec = {0};
  spec.pooled_height = getOutputHeight();
  spec.pooled_width = getOutputWidth();
  spec.sampling_ratio = getSamplingRatio();
  spec.spatial_scale = getSpatialScale().convertToDouble();
  // spec.align_corners = false;
  spec.align_corners = getAlignCorners();
  spec.position_sensitive = false;
  spec.plat_sp = 0;
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::RoiAlignOp::get_fw_type_bm1684x() { return FW_BMNET_ROI_ALIGN; }

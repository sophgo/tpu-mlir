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
void tpu::RoiExtractorOp::codegen_global_bm1684x() {
  UNREACHABLE_THIS("Only support dynamic codegen");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::RoiExtractorOp::dyn_codegen_global_bm1684x(void *buffer) {
  assert(getMode() != "Max");
  if (!buffer) {
    return sizeof(roi_extractor_spec_t);
  }
  roi_extractor_spec_t spec = {0};
  spec.num_levels = getNumLevels();
  assert(spec.num_levels >= 1 & spec.num_levels <= MAX_ROI_ALIGN_NUM_LEVELS);
  spec.pooled_height = getOutputHeight();
  spec.pooled_width = getOutputWidth();
  spec.sampling_ratio = getSamplingRatio();
  assert(spec.sampling_ratio > 0);

  auto spatial_scales = module::getF64Array(getSpatialScales());
  for (int i = 0; i < spec.num_levels; i++) {
    spec.spatial_scales[i] = spatial_scales->at(i);
  }

  spec.align_corners = getAlignCorners();
  spec.position_sensitive = false;
  spec.plat_sp = 0;
  assert(spec.position_sensitive == false);
  spec.isStatic = getIsStatic();
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::RoiExtractorOp::get_fw_type_bm1684x() {
  return FW_BMNET_ROI_EXTRACTOR;
}

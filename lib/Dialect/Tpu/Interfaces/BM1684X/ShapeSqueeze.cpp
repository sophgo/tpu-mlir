//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynCompileCommon.hpp"
#include "tpu_mlir/Support/MathUtils.h"
using namespace tpu_mlir::backend;

// ======================================
// GlobalGenInterface
// ======================================
void tpu::ShapeSqueezeOp::codegen_global_bm1684x() {
  llvm_unreachable("Not supported now");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ShapeSqueezeOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(squeeze_dims_common_spec_t);
  squeeze_dims_common_spec_t param = {0};
  const auto axes = module::getI64Array(getAxes());
  param.axis_num = axes->size();
  for (int i = 0; i < param.axis_num; i++) {
    param.axis_list[i] = axes->at(i);
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::ShapeSqueezeOp::get_fw_type_bm1684x() {
  return FW_BMNET_SHAPE_SQUEEZE;
}

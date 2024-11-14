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
void tpu::UnsqueezeOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  tile_1d_global_param_t param = {0};
  param.tile_axis = 0;
  param.tile_num = 1;
  param.type = 0;
  BM168x::call_global_func("backend_api_tile_1d_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::UnsqueezeOp::dyn_codegen_global_bm1684x(void *buffer) {
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

int64_t tpu::UnsqueezeOp::get_fw_type_bm1684x() { return FW_BMNET_UNSQUEEZE; }

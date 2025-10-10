//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
// =========================================
// GlobalGenInterface
// =========================================

using namespace tpu_mlir::backend;

void tpu::ScaleLutOp::codegen_global_bm1684x() {
  auto op = getOperation();
  scalelut_param_t param = {0};
  param.shape_dim = 4;
  param.table_length = 256;
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_scalelut_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ScaleLutOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type,
    bool with_hw_margins) {
  return 0;
}

void tpu::ScaleLutOp::codegen_local_bm1684x_kernel(
    std::vector<group_info_t> &in_group_infos,
    std::vector<group_info_t> &out_group_infos, local_sec_info_t &sec_info,
    std::shared_ptr<std::vector<tensor_spec_t>> input_spec,
    std::shared_ptr<std::vector<tensor_spec_t>> output_spec) {
  UNREACHABLE_THIS("Not Implemented");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================

int64_t tpu::ScaleLutOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(scalelut_param_t);
  scalelut_param_t param = {0};
  param.shape_dim = 4;
  param.table_length = 256;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::ScaleLutOp::get_fw_type_bm1684x() { return FW_BMNET_SCALE_LUT; }

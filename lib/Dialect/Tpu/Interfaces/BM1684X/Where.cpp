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

void tpu::WhereOp::codegen_global_bm1684x() {
  select_common_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.sel0_is_const = getXIsConst();
  spec.sel1_is_const = getYIsConst();
  spec.sel0_const_val = getXConstVal().convertToDouble();
  spec.sel1_const_val = getYConstVal().convertToDouble();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_select_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::WhereOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type,
    bool with_hw_margins) {
  return 0;
}

void tpu::WhereOp::codegen_local_bm1684x_kernel(
    std::vector<group_info_t> &in_group_infos,
    std::vector<group_info_t> &out_group_infos, local_sec_info_t &sec_info,
    std::shared_ptr<std::vector<tensor_spec_t>> input_spec,
    std::shared_ptr<std::vector<tensor_spec_t>> output_spec) {
  select_common_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.sel0_is_const = getXIsConst();
  spec.sel1_is_const = getYIsConst();
  spec.sel0_const_val = getXConstVal().convertToDouble();
  spec.sel1_const_val = getYConstVal().convertToDouble();

  BM168x::call_local_func("backend_api_select_local", &spec, sizeof(spec),
                          &sec_info, input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::WhereOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(select_common_spec_t);
  select_common_spec_t spec = {0};
  spec.sel0_is_const = getXIsConst();
  spec.sel1_is_const = getYIsConst();
  spec.sel0_const_val = getXConstVal().convertToDouble();
  spec.sel1_const_val = getYConstVal().convertToDouble();
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::WhereOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(select_common_spec_t);
  select_common_spec_t spec = {0};
  spec.sel0_is_const = getXIsConst();
  spec.sel1_is_const = getYIsConst();
  spec.sel0_const_val = getXConstVal().convertToDouble();
  spec.sel1_const_val = getYConstVal().convertToDouble();
  spec.buffer_addr = module::getAddress(getBuffer());
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::WhereOp::get_fw_type_bm1684x() { return FW_BMNET_SELECT; }

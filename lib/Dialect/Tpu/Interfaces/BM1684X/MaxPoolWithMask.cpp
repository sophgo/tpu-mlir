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

static void SpecAssign(const pool_attr_t &attr, pooling_common_spec_t &spec) {
  spec.kh = attr.kh;
  spec.kw = attr.kw;
  spec.pad_h_t = attr.pad_h;
  spec.pad_h_b = attr.pad_h_after;
  spec.pad_w_l = attr.pad_w;
  spec.pad_w_r = attr.pad_w_after;
  spec.stride_h = attr.sh;
  spec.stride_w = attr.sw;
  spec.dh = 1;
  spec.dw = 1;
  spec.is_global_pooling = attr.is_global;
  spec.avg_pooling_mode = 0;
  spec.if_relu = attr.do_relu;
  spec.relu_limit = attr.relu_limit;
  spec.max_pooling_with_mask = true;
  spec.is_avg_pooling = false;
  spec.ceil_mode = 0;
  spec.round_mode = ROUND_UP;
}

// =========================================
// GlobalGenInterface
// =========================================

void tpu::MaxPoolWithMaskOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto attr = parseParam();
  pooling_common_spec_t spec = {0};
  SpecAssign(attr, spec);
  BM168x::call_global_func("backend_api_pooling_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::MaxPoolWithMaskOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return 0;
}

void tpu::MaxPoolWithMaskOp::codegen_local_bm1684x(
    int64_t n_step, int64_t c_step, int64_t h_step, int64_t d_step,
    int64_t w_step, group_type_t group_type, local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step,
                                               d_step, w_step, c_step);

  auto attr = parseParam();
  pooling_local_spec_t spec = {0};
  auto &common = spec.common;
  SpecAssign(attr, common);
  spec.buffer_addr = gi.buffer_addr;
  common.pad_h_t = (in_gi.h_idx == 0 ? attr.pad_h : 0);
  common.pad_h_b =
      (in_gi.h_idx + in_gi.h_slice == attr.ih ? attr.pad_h_after : 0);
  common.pad_w_l = (in_gi.w_idx == 0 ? attr.pad_w : 0);
  common.pad_w_r =
      (in_gi.w_idx + in_gi.w_slice == attr.iw ? attr.pad_w_after : 0);

  BM168x::call_local_func("backend_api_pooling_local", &spec, sizeof(spec),
                          &sec_info, input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::MaxPoolWithMaskOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(pooling_local_spec_t);
  auto gi = getGroupInfo(0, 0, 0, 0, 0);
  auto attr = parseParam();
  pooling_local_spec_t spec = {0};
  auto &common = spec.common;
  SpecAssign(attr, common);
  spec.buffer_addr = gi.buffer_addr;
  common.pad_h_t = attr.pad_h;
  common.pad_h_b = attr.pad_h_after;
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::MaxPoolWithMaskOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(pooling_common_spec_t);
  auto attr = parseParam();
  pooling_common_spec_t spec = {0};
  SpecAssign(attr, spec);
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::MaxPoolWithMaskOp::get_fw_type_bm1684x() { return FW_BMNET_POOL; }

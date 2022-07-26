//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Dnnl/Pool.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pooling_common_spec {
  int32_t kh;
  int32_t kw;
  int32_t pad_h_t;
  int32_t pad_h_b;
  int32_t pad_w_l;
  int32_t pad_w_r;
  int32_t stride_h;
  int32_t stride_w;
  int32_t dh;
  int32_t dw;
  int32_t is_global_pooling;
  int32_t is_avg_pooling;
  int32_t avg_pooling_mode;
  /* for float */
  int32_t if_relu;
  float relu_upper_limit;
  /* for fix8b */
  int32_t ceil_mode;
  int32_t round_mode;
  int32_t avg_pooling_quant_mode;
  int32_t max_pooling_with_mask; //1: with mask 0: no mask
  int32_t multiplier;
  int32_t rshiftbits;
  /* asymmetric quantize */
  int32_t merge_requant;
  float   rq_scale;
  float   rq_offset;
} pooling_common_spec_t;

typedef struct {
  int32_t buffer_addr;
  pooling_common_spec_t common;
} pooling_local_spec_t;

#ifdef __cplusplus
}
#endif

static void SpecAssign(const pool_attr_t &attrs, pooling_common_spec_t &spec) {
  spec.kh = attrs.kh;
  spec.kw = attrs.kw;
  spec.pad_h_t = attrs.pad_h;
  spec.pad_h_b = attrs.pad_h_after;
  spec.pad_w_l = attrs.pad_w;
  spec.pad_w_r = attrs.pad_w_after;
  spec.stride_h = attrs.sh;
  spec.stride_w = attrs.sw;
  spec.dh = 1;
  spec.dw = 1;
  spec.is_global_pooling = attrs.is_global;
  spec.avg_pooling_mode = attrs.count_include_pad ? 0 : 1;
  spec.if_relu = attrs.do_relu;
  spec.relu_upper_limit = -1.;
  spec.ceil_mode = 0;
  spec.round_mode = ROUND_UP;
  /// TODO: may be need support pad value for pooling2D and pooling 3D
  /// spec.pad_value = attrs.pad_value;
}

// =========================================
// GlobalGenInterface
// =========================================
// int8
void tpu::MaxPool2DOp::codegen_global_int8_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  pooling_common_spec_t spec = {0};
  pool_attr_t attrs;
  parseParam(&attrs);
  SpecAssign(attrs, spec);
  spec.is_avg_pooling = false;
  BM1684x::instance().call_global_func("backend_api_pooling_global", &spec,
                                       sizeof(spec), input_spec->data(),
                                       output_spec->data());
}

void tpu::AvgPool2DOp::codegen_global_int8_bm1684x() {
  auto op = getOperation();
  auto module = Module::getModuleOp(op);
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  pool_attr_t attrs;
  parseParam(&attrs);
  pooling_common_spec_t spec = {0};
  SpecAssign(attrs, spec);
  spec.is_avg_pooling = true;
  spec.avg_pooling_quant_mode = Module::getAsymmetric(module) ? 2 : 0;
  if (spec.avg_pooling_quant_mode == 0) {
    spec.multiplier = multiplier().getValue();
    spec.rshiftbits = rshift().getValue();
  } else if (spec.avg_pooling_quant_mode == 2) {
    spec.merge_requant = true;
    spec.rq_scale = scale().getValue().convertToDouble();
    spec.rq_offset = offset().getValue().convertToDouble();
  }
  BM1684x::instance().call_global_func("backend_api_pooling_global", &spec,
                                       sizeof(spec), input_spec->data(),
                                       output_spec->data());
}

// f32
void tpu::AvgPool2DOp::codegen_global_float_bm1684x() {
  pool_attr_t attrs;
  parseParam(&attrs);
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  pooling_common_spec_t spec = {0};
  SpecAssign(attrs, spec);
  spec.is_avg_pooling = true;
  BM1684x::instance().call_global_func("backend_api_pooling_global", &spec,
                                       sizeof(spec), input_spec->data(),
                                       output_spec->data());
}

void tpu::MaxPool2DOp::codegen_global_float_bm1684x() {
  pool_attr_t attrs;
  parseParam(&attrs);
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  pooling_common_spec_t spec = {0};
  SpecAssign(attrs, spec);
  spec.is_avg_pooling = false;
  BM1684x::instance().call_global_func("backend_api_pooling_global", &spec,
                                       sizeof(spec), input_spec->data(),
                                       output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::AvgPool2DOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t size = 0;
  auto module = Module::getModuleOp(getOperation());
  if (Module::getAsymmetric(module)) {
    auto kernel = Module::getI64Array(kernel_shape());
    int64_t dtype_bytes =
        kernel->at(0) * kernel->at(1) > 256 ? sizeof(int) : sizeof(short);
    int64_t eu_num = BM1684x::instance().get_eu_num(dtype_bytes);
    int64_t npu_num = BM1684x::instance().get_npu_num();

    int64_t N, C, H, W;
    Module::getNCHW(input(), N, C, H, W);
    size = align_up(out_hslice * W, eu_num) * ceiling_func(C, npu_num) *
           dtype_bytes;
  }
  return size;
}

int64_t tpu::MaxPool2DOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::MaxPool2DOp::codegen_local_int8_bm1684x(int64_t n_step,
                                                  int64_t h_step) {
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);

  pool_attr_t attrs;
  parseParam(&attrs);
  pooling_local_spec_t spec = {0};
  auto &common = spec.common;
  SpecAssign(attrs, common);
  common.is_avg_pooling = false;
  common.pad_h_t = (in_gi.h_idx == 0 ? attrs.pad_h : 0);
  common.pad_h_b =
      (in_gi.h_idx + in_gi.h_slice == attrs.ih ? attrs.pad_h_after : 0);

  local_sec_info_t sec_info;
  memset(&sec_info, 0, sizeof(sec_info));
  sec_info.n_slice = in_gi.n_slice;
  sec_info.h_slice = in_gi.h_slice;
  sec_info.h_idx = in_gi.h_idx;
  sec_info.is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == attrs.ih);
  // to be compatible with nntoolchain
  if (sec_info.is_h_split) {
    sec_info.h_idx = h_step == 0 ? -attrs.pad_h : in_gi.h_idx;
    sec_info.h_slice = sec_info.h_idx < 0 ? sec_info.h_slice - sec_info.h_idx
                                          : sec_info.h_slice;
    sec_info.h_slice = sec_info.h_slice + common.pad_h_b;
  }
  sec_info.w_slice = attrs.iw;
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_slice = attrs.ow;
  BM1684x::instance().call_local_func("backend_api_pooling_local", &spec,
                                      sizeof(spec), &sec_info,
                                      input_spec->data(), output_spec->data());
}

void tpu::AvgPool2DOp::codegen_local_int8_bm1684x(int64_t n_step,
                                                  int64_t h_step) {
  auto op = getOperation();
  auto module = Module::getModuleOp(op);
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);

  pool_attr_t attrs;
  parseParam(&attrs);
  pooling_local_spec_t spec = {0};
  auto &common = spec.common;
  SpecAssign(attrs, common);
  spec.buffer_addr = gi.buffer_addr;
  common.pad_h_t = (in_gi.h_idx == 0 ? attrs.pad_h : 0);
  common.pad_h_b =
      (in_gi.h_idx + in_gi.h_slice == attrs.ih ? attrs.pad_h_after : 0);
  common.is_avg_pooling = true;
  common.avg_pooling_quant_mode = Module::getAsymmetric(module) ? 2 : 0;
  if (common.avg_pooling_quant_mode == 0) {
    common.multiplier = multiplier().hasValue() ? multiplier().getValue() : -1;
    common.rshiftbits = rshift().hasValue() ? rshift().getValue() : -1;
  } else if (common.avg_pooling_quant_mode == 2) {
    common.merge_requant = true;
    common.rq_scale =
        scale().hasValue() ? (scale().getValue().convertToDouble()) : -1.;
    common.rq_offset =
        offset().hasValue() ? (offset().getValue().convertToDouble()) : -1.;
  }

  local_sec_info_t sec_info;
  memset(&sec_info, 0, sizeof(sec_info));
  sec_info.n_slice = in_gi.n_slice;
  sec_info.h_slice = in_gi.h_slice;
  sec_info.h_idx = in_gi.h_idx;
  sec_info.is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == attrs.ih);
  // to be compatible with nntoolchain
  if (sec_info.is_h_split) {
    sec_info.h_idx = h_step == 0 ? -attrs.pad_h : in_gi.h_idx;
    sec_info.h_slice = sec_info.h_idx < 0 ? sec_info.h_slice - sec_info.h_idx
                                          : sec_info.h_slice;
    sec_info.h_slice = sec_info.h_slice + common.pad_h_b;
  }
  sec_info.w_slice = attrs.iw;
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_slice = attrs.ow;
  BM1684x::instance().call_local_func("backend_api_pooling_local", &spec,
                                      sizeof(spec), &sec_info,
                                      input_spec->data(), output_spec->data());
}

void tpu::MaxPool2DOp::codegen_local_float_bm1684x(int64_t n_step,
                                                   int64_t h_step) {
  codegen_local_int8_bm1684x(n_step, h_step);
}

void tpu::AvgPool2DOp::codegen_local_float_bm1684x(int64_t n_step,
                                                   int64_t h_step) {
  codegen_local_int8_bm1684x(n_step, h_step);
}

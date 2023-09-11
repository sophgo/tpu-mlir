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

// ======================================
// GlobalGenInterface
// ======================================

void tpu::Conv2DOp::codegen_global_bm1684x() {
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (attr.dims == 1) {
    BM168x::fix_shape(input_spec->at(0), {attr.n, attr.ic, attr.ih, attr.iw});
    BM168x::fix_shape(output_spec->at(0), {attr.n, attr.oc, attr.oh, attr.ow});
  }
  conv_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  auto &common = spec.common;
  common.input_c = attr.ic;
  common.output_c = attr.oc;
  common.if_relu = attr.do_relu;
  common.upper_limit = attr.relu_limit;
  common.kh = attr.kh;
  common.kw = attr.kw;
  common.dh = attr.dh;
  common.dw = attr.dw;
  common.stride_h = attr.sh;
  common.stride_w = attr.sw;
  common.groups = attr.groups;
  common.pad_h_t = attr.pht;
  common.pad_h_b = attr.phb;
  common.pad_w_l = attr.pwl;
  common.pad_w_r = attr.pwr;
  common.round_mode = ROUNDING_HALF_UP;
  common.has_bias = attr.has_bias;
  common.bias_sign = true;
  common.ipad_is_const = true;
  common.kzp_is_const = true;
  common.kzp_value = attr.kernel_zp;
  common.use_3ic_optimize = getUse_3icOptimize();
  common.weight_is_coeff = attr.weight_is_coeff;
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    auto out_etype = module::getStorageType(getOutput());
    if (out_etype.isUnsignedInteger()) {
      common.if_relu = true;
    }
    if (out_etype.isInteger(32)) {
      bool coeff_merge = getCoeffMerged();
      spec.merge_coeff = coeff_merge ? 1 : 0;
    } else {
      spec.merge_coeff = 2;
    }
    common.is_asym = true;
    common.ipad_value = in_qtype.getZeroPoint();
  }
  BM168x::call_global_func("backend_api_conv_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::Conv2DOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {

  auto &p = getConv2DParam(*this);
  int64_t sz = 0;
  auto in_type = BM168x::getDataType(getInput());
  auto out_type = BM168x::getDataType(getOutput());
  auto in_type_len = BM168x::getFmtBytes(in_type);
  auto out_type_len = BM168x::getFmtBytes(out_type);
  auto eu_num = BM168x::eu_num(in_type_len);
  int oc_per_npu = ceiling_func(p.oc, BM168x::NPU_NUM);
  int ic_per_npu = ceiling_func(p.ic / p.groups, BM168x::NPU_NUM);
  int int32_size = out_lmem_bytes * sizeof(int32_t) / out_type_len;
  int use_3ic_optimize = getUse_3icOptimize();
  if (module::isBM1686() && getCoeffMerged()) {
    if (module::isUniformQuantized(getInput()) && p.kernel_zp != 0)
      return int32_size * 2;
    if(use_3ic_optimize == 0)
      return 0;
  }
  if (getCoeffMerged()) {
    sz += int32_size;
  }
  if (p.groups > 1) {
    sz += in_nslice * ic_per_npu * align_up(in_hslice * in_wslice, eu_num) *
          in_type_len;
    sz += ic_per_npu * 2 * in_type_len;
  }

  if (p.is_dw) {
    sz += int32_size;
    sz += oc_per_npu * p.kh * p.kw;
  }

  if (use_3ic_optimize & 0x20) {
    // used for broadcast input
    sz += in_lmem_bytes;
  }
  int use_3ic = (use_3ic_optimize & 0x3);
  if (use_3ic == 1) { // merge kh to ic
    sz += align_up(out_hslice * in_wslice, eu_num) * in_nslice * in_type_len;
    sz += 64 * 2;
    sz += p.kh * in_type_len;
  } else if (use_3ic == 2) { // merge kw to ic
    sz += align_up(in_hslice * out_wslice, eu_num) * in_nslice * in_type_len;
    sz += 64 * 2;
    sz += p.kw * in_type_len;
  } else if (use_3ic == 3) { // merge kh and kw to ic
    sz += align_up(out_hslice * out_wslice, eu_num) * in_nslice * in_type_len;
    sz += 64 * 2;
    sz += p.kh * p.kw * in_type_len;
  }
  auto Filter_type = BM168x::getDataType(getFilter());
  auto Filter_type_len = BM168x::getFmtBytes(Filter_type);
  if ((module::isWeight(getFilter()) == false)) {
    if (Filter_type == DTYPE_FP16 || Filter_type == DTYPE_BFP16 || Filter_type == DTYPE_INT8 || Filter_type == DTYPE_UINT8){
      sz += p.kh * p.kw * p.oc * align_up(p.ic / p.groups, eu_num) * Filter_type_len;
    }
  }
  return sz;
}

void tpu::Conv2DOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                          int64_t h_step, int64_t d_step,
                                          int64_t w_step,
                                          group_type_t group_type,
                                          local_sec_info_t &sec_info) {
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (attr.dims == 1) {
    BM168x::fix_shape(input_spec->at(0), {attr.n, attr.ic, attr.ih, attr.iw});
    BM168x::fix_shape(output_spec->at(0), {attr.n, attr.oc, attr.oh, attr.ow});
  }
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step,
                                               d_step, w_step, c_step);

  conv_local_param_t p;
  memset(&p, 0, sizeof(p));
  p.spec.buffer_local_addr = gi.buffer_addr;
  auto &common = p.spec.common;
  common.input_c = attr.ic;
  common.output_c = attr.oc;
  common.if_relu = attr.do_relu;
  common.upper_limit = attr.relu_limit;
  common.kh = attr.kh;
  common.kw = attr.kw;
  common.dh = attr.dh;
  common.dw = attr.dw;
  common.stride_h = attr.sh;
  common.stride_w = attr.sw;
  common.groups = attr.groups;
  common.pad_h_t = (in_gi.h_idx == 0 ? attr.pht : 0);
  common.pad_h_b = (in_gi.h_idx + in_gi.h_slice == attr.ih ? attr.phb : 0);
  common.pad_w_l = (in_gi.w_idx == 0 ? attr.pwl : 0);
  common.pad_w_r = (in_gi.w_idx + in_gi.w_slice == attr.iw ? attr.pwr : 0);
  common.round_mode = ROUNDING_HALF_UP;
  common.has_bias = attr.has_bias;
  common.bias_sign = true;
  common.ipad_is_const = true;
  common.kzp_is_const = true;
  common.kzp_value = attr.kernel_zp;
  common.use_3ic_optimize = getUse_3icOptimize();
  common.weight_is_coeff = attr.weight_is_coeff;

  p.spec.unused_ht_for_input = 0;
  p.spec.unused_hb_for_input = 0;
  p.spec.unused_wl_for_input = 0;
  p.spec.unused_wr_for_input = 0;
  int64_t N, C, H, W;
  module::getNCHW(getInput(), N, C, H, W);
  if (sec_info.h_slice != H) {
    int cal_h_idx = sec_info.out_h_idx * attr.sh - attr.pht;
    int cal_h_slice = (sec_info.out_h_slice - 1) * attr.sh + attr.kh;
    cal_h_slice = std::min(cal_h_slice, cal_h_slice + cal_h_idx);
    cal_h_idx = std::max(0, cal_h_idx);
    p.spec.unused_ht_for_input = cal_h_idx - std::max(0, sec_info.h_idx);
    int h_end = std::min(sec_info.h_idx + sec_info.h_slice, (int)H);
    p.spec.unused_hb_for_input = std::max(0, h_end - (cal_h_idx + cal_h_slice));
  }

  if (sec_info.w_slice != W) {
    int cal_w_idx = sec_info.out_w_idx * attr.sw - attr.pwl;
    int cal_w_slice = (sec_info.out_w_slice - 1) * attr.sw + attr.kw;
    cal_w_slice = std::min(cal_w_slice, cal_w_slice + cal_w_idx);
    cal_w_idx = std::max(0, cal_w_idx);

    p.spec.unused_wl_for_input = cal_w_idx - std::max(0, sec_info.w_idx);
    int w_end = std::min(sec_info.w_idx + sec_info.w_slice, (int)W);
    p.spec.unused_wr_for_input = std::max(0, w_end - (cal_w_idx + cal_w_slice));
  }

  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    common.ipad_value = in_qtype.getZeroPoint();
    common.is_asym = true;
    auto out_etype = module::getStorageType(getOutput());
    common.if_relu = out_etype.isUnsignedInteger();
    if (out_etype.isInteger(32)) {
      bool coeff_merge = getCoeffMerged();
      p.spec.merge_coeff = coeff_merge ? 1 : 0;
      p.spec.with_requant = 0;
    } else {
      p.spec.merge_coeff = 2;
      p.spec.with_requant = 1;
    }
  }
  BM168x::call_local_func("backend_api_conv_local", &p, sizeof(p), &sec_info,
                          input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::Conv2DOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(conv_local_param_t);
  conv_local_param_t param;
  memset(&param, 0, sizeof(param));
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (attr.dims == 1) {
    BM168x::fix_shape(input_spec->at(0), {attr.n, attr.ic, attr.ih, attr.iw});
    BM168x::fix_shape(output_spec->at(0), {attr.n, attr.oc, attr.oh, attr.ow});
  }
  auto gi = getGroupInfo(0, 0, 0, 0, 0);

  param.spec.buffer_local_addr = gi.buffer_addr;
  auto &common = param.spec.common;
  common.input_c = attr.ic;
  common.output_c = attr.oc;
  common.if_relu = attr.do_relu;
  common.upper_limit = attr.relu_limit;
  common.kh = attr.kh;
  common.kw = attr.kw;
  common.dh = attr.dh;
  common.dw = attr.dw;
  common.stride_h = attr.sh;
  common.stride_w = attr.sw;
  common.groups = attr.groups;
  common.pad_h_t = attr.pht;
  common.pad_h_b = attr.phb;
  common.pad_w_l = attr.pwl;
  common.pad_w_r = attr.pwr;
  common.round_mode = ROUNDING_HALF_UP;
  common.has_bias = attr.has_bias;
  common.bias_sign = true;
  common.ipad_is_const = true;
  common.kzp_is_const = true;
  common.kzp_value = attr.kernel_zp;
  common.use_3ic_optimize = getUse_3icOptimize();
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    common.ipad_value = in_qtype.getZeroPoint();
    common.is_asym = true;
    auto out_etype = module::getStorageType(getOutput());
    common.if_relu = out_etype.isUnsignedInteger();
    if (out_etype.isInteger(32)) {
      bool coeff_merge = getCoeffMerged();
      param.spec.merge_coeff = coeff_merge ? 1 : 0;
      param.spec.with_requant = 0;
    } else {
      param.spec.merge_coeff = 2;
      param.spec.with_requant = 1;
    }
  }
  param.spec.reference_id = get_tensor_id(op->getResult(0));
  param.spec.concat_c = attr.oc;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::Conv2DOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(conv_global_spec_t);
  conv_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  auto attr = parseParam();
  auto &common = spec.common;
  common.input_c = attr.ic;
  common.output_c = attr.oc;
  common.if_relu = attr.do_relu;
  common.upper_limit = attr.relu_limit;
  common.kh = attr.kh;
  common.kw = attr.kw;
  common.dh = attr.dh;
  common.dw = attr.dw;
  common.stride_h = attr.sh;
  common.stride_w = attr.sw;
  common.groups = attr.groups;
  common.pad_h_t = attr.pht;
  common.pad_h_b = attr.phb;
  common.pad_w_l = attr.pwl;
  common.pad_w_r = attr.pwr;
  common.round_mode = ROUNDING_HALF_UP;
  common.has_bias = attr.has_bias;
  common.bias_sign = true;
  common.ipad_is_const = true;
  common.kzp_is_const = true;
  common.kzp_value = attr.kernel_zp;
  common.use_3ic_optimize = getUse_3icOptimize();
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    common.ipad_value = in_qtype.getZeroPoint();
    common.is_asym = true;
    auto out_etype = module::getStorageType(getOutput());
    common.if_relu = out_etype.isUnsignedInteger();
    if (out_etype.isInteger(32)) {
      bool coeff_merge = getCoeffMerged();
      spec.merge_coeff = coeff_merge ? 1 : 0;
    } else {
      spec.merge_coeff = 2;
    }
  }
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::Conv2DOp::get_fw_type_bm1684x() { return FW_BMNET_CONV; }

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

void tpu::Conv3DOp::codegen_global_bm1684x() {
  auto attr = parseParam();
  conv3d_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.input_global_addr = module::getAddress(getInput());
  spec.weight_global_addr = module::getAddress(getFilter());
  spec.output_global_addr = module::getAddress(getOutput());
  if (attr.has_bias) {
    spec.has_bias = 1;
    spec.bias_global_addr = module::getAddress(getBias());
    spec.bias_dtype = BM168x::getDataType(getBias());
  }
  auto shape = module::getShape(getInput());
  for (size_t i = 0; i < shape.size(); ++i) {
    spec.input_shape[i] = shape[i];
  }
  spec.groups = attr.groups;
  spec.output_c = attr.oc;
  spec.kernel[0] = attr.kd;
  spec.kernel[1] = attr.kh;
  spec.kernel[2] = attr.kw;
  spec.stride[0] = attr.sd;
  spec.stride[1] = attr.sh;
  spec.stride[2] = attr.sw;
  spec.dilation[0] = attr.dd;
  spec.dilation[1] = attr.dh;
  spec.dilation[2] = attr.dw;
  spec.pad[0] = attr.pdf;
  spec.pad[1] = attr.pdb;
  spec.pad[2] = attr.pht;
  spec.pad[3] = attr.phb;
  spec.pad[4] = attr.pwl;
  spec.pad[5] = attr.pwr;
  spec.input_dtype = BM168x::getDataType(getInput());
  spec.weight_dtype = BM168x::getDataType(getFilter());
  spec.output_dtype = BM168x::getDataType(getOutput());
  spec.do_relu = attr.do_relu;
  spec.relu_limit = attr.relu_limit;
  if (module::getStorageType(getInput()).isIntOrIndex()) {
    auto out_etype = module::getStorageType(getOutput());
    spec.do_relu = out_etype.isUnsignedInteger();
    if (module::isUniformQuantized(getInput())) {
      auto in_qtype = module::getUniformQuantizedType(getInput());
      spec.pad_val = in_qtype.getZeroPoint();
    }
    spec.kzp_is_const = true;
    spec.kzp_val = attr.kernel_zp;
    spec.kzp_dtype = spec.weight_dtype;
    spec.pad_is_const = true;
  }
  BM168x::call_global_func("backend_api_conv3d_global", &spec, sizeof(spec));
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::Conv3DOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  auto attr = parseParam();
  int64_t sz = 0;
  int64_t npu_num = BM168x::NPU_NUM;

  int32_t oc_per_npu = 0;
  for (int32_t i = 0; i < attr.groups; i++) {
    oc_per_npu =
        std::max((int64_t)oc_per_npu,
                 ceiling_func(i * attr.groups % npu_num + attr.oc / attr.groups,
                              npu_num));
  }
  auto in_type = module::getStorageType(getInput().getType());
  auto out_type = module::getStorageType(getOutput().getType());
  // output start npu id must be same with weight start npu id
  if ((in_type.isF16() || in_type.isBF16()) && !out_type.isF32() &&
      attr.kd > 1) {
    sz += (oc_per_npu *
           align_up(out_hslice * out_wslice, BM168x::eu_num(sizeof(float))) *
           sizeof(float));
  }

  // input must start from npu 0
  if ((in_type.isF16() || in_type.isBF16()) && attr.groups > 1) {
    sz += ceiling_func((int64_t)attr.ic / attr.groups, npu_num) *
          align_up(in_hslice * in_wslice, BM168x::eu_num(sizeof(int16_t))) *
          sizeof(int16_t);
  }
  if ((in_type.isInteger(8)) && attr.groups > 1) {
    sz += ceiling_func((int64_t)attr.ic / attr.groups, npu_num) *
          align_up(in_hslice * in_wslice, BM168x::eu_num(sizeof(int8_t))) *
          sizeof(int8_t);
  }
  return sz;
}

void tpu::Conv3DOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                          int64_t h_step, int64_t d_step,
                                          int64_t w_step,
                                          group_type_t group_type,
                                          local_sec_info_t &sec_info) {
  auto attr = parseParam();
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step,
                                               d_step, w_step, c_step);

  conv3d_local_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.input_local_addr = in_gi.out_addr;
  spec.weight_local_addr =
      LocalGenInterface::getGroupInfo(getFilter()).out_addr;
  if (attr.has_bias) {
    spec.has_bias = true;
    spec.bias_local_addr = LocalGenInterface::getGroupInfo(getBias()).out_addr;
    spec.bias_dtype = BM168x::getDataType(getBias());
  }
  spec.buffer_local_addr = gi.buffer_addr;
  spec.output_local_addr = gi.out_addr;
  spec.input_shape[0] = sec_info.d_slice;
  spec.input_shape[1] = sec_info.n_slice;
  spec.input_shape[2] = attr.ic;
  spec.input_shape[3] = sec_info.h_slice;
  spec.input_shape[4] = sec_info.w_slice;
  spec.groups = attr.groups;
  spec.output_c = attr.oc;
  spec.kernel[0] = attr.kd;
  spec.kernel[1] = attr.kh;
  spec.kernel[2] = attr.kw;
  spec.stride[0] = attr.sd;
  spec.stride[1] = attr.sh;
  spec.stride[2] = attr.sw;
  spec.dilation[0] = attr.dd;
  spec.dilation[1] = attr.dh;
  spec.dilation[2] = attr.dw;
  spec.pad[0] = in_gi.d_idx == 0 ? attr.pdf : 0;
  spec.pad[1] = in_gi.d_idx + in_gi.d_slice >= attr.id ? attr.pdb : 0;
  spec.pad[2] = in_gi.h_idx == 0 ? attr.pht : 0;
  spec.pad[3] = in_gi.h_idx + in_gi.h_slice >= attr.ih ? attr.phb : 0;
  spec.pad[4] = in_gi.w_idx == 0 ? attr.pwl : 0;
  spec.pad[5] = in_gi.w_idx + in_gi.w_slice >= attr.iw ? attr.pwr : 0;
  spec.input_dtype = BM168x::getDataType(getInput());
  spec.weight_dtype = BM168x::getDataType(getFilter());
  spec.output_dtype = BM168x::getDataType(getOutput());
  spec.do_relu = attr.do_relu;
  spec.relu_limit = attr.relu_limit;
  if (module::getStorageType(getInput()).isIntOrIndex()) {
    auto out_etype = module::getStorageType(getOutput());
    spec.do_relu = out_etype.isUnsignedInteger(8);
    if (module::isUniformQuantized(getInput())) {
      auto in_qtype = module::getUniformQuantizedType(getInput());
      spec.pad_val = in_qtype.getZeroPoint();
    }
    spec.kzp_is_const = true;
    spec.kzp_val = attr.kernel_zp;
    spec.kzp_dtype = spec.weight_dtype;
    spec.pad_is_const = true;
  }
  BM168x::call_local_func("backend_api_conv3d_local", &spec,
                          sizeof(conv3d_local_spec_t));
}

// dynamic codegen
int64_t tpu::Conv3DOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_conv3d_local_param_t);
  auto attr = parseParam();
  auto gi = getGroupInfo(0, 0, 0, 0, 0);

  dyn_conv3d_local_param_t param = {0};
  if (attr.has_bias) {
    param.spec.common.has_bias = true;
    param.spec.common.bias_dtype = BM168x::getDataType(getBias());
  }
  param.spec.buffer_local_addr = gi.buffer_addr;
  param.spec.common.groups = attr.groups;
  param.spec.common.output_c = attr.oc;
  param.spec.common.kernel[0] = attr.kd;
  param.spec.common.kernel[1] = attr.kh;
  param.spec.common.kernel[2] = attr.kw;
  param.spec.common.stride[0] = attr.sd;
  param.spec.common.stride[1] = attr.sh;
  param.spec.common.stride[2] = attr.sw;
  param.spec.common.dilation[0] = attr.dd;
  param.spec.common.dilation[1] = attr.dh;
  param.spec.common.dilation[2] = attr.dw;
  param.spec.common.pad[0] = attr.pdf;
  param.spec.common.pad[1] = attr.pdb;
  param.spec.common.pad[2] = attr.pht;
  param.spec.common.pad[3] = attr.phb;
  param.spec.common.pad[4] = attr.pwl;
  param.spec.common.pad[5] = attr.pwr;
  param.spec.common.input_dtype = BM168x::getDataType(getInput());
  param.spec.common.weight_dtype = BM168x::getDataType(getFilter());
  param.spec.common.output_dtype = BM168x::getDataType(getOutput());
  param.spec.common.do_relu = attr.do_relu;
  param.spec.common.relu_limit = attr.relu_limit;
  if (module::getStorageType(getInput()).isIntOrIndex()) {
    auto out_etype = module::getStorageType(getOutput());
    param.spec.common.do_relu = out_etype.isUnsignedInteger(8);
    if (module::isUniformQuantized(getInput())) {
      auto in_qtype = module::getUniformQuantizedType(getInput());
      param.spec.common.pad_val = in_qtype.getZeroPoint();
    }
    param.spec.common.kzp_is_const = true;
    param.spec.common.kzp_val = attr.kernel_zp;
    param.spec.common.kzp_dtype = param.spec.common.weight_dtype;
    param.spec.common.pad_is_const = true;
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::Conv3DOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_conv3d_global_param_t);
  auto attr = parseParam();
  dyn_conv3d_global_param_t param = {0};
  if (attr.has_bias) {
    param.spec.common.has_bias = 1;
    param.spec.common.bias_dtype = BM168x::getDataType(getBias());
  }

  param.spec.common.groups = attr.groups;
  param.spec.common.output_c = attr.oc;
  param.spec.common.kernel[0] = attr.kd;
  param.spec.common.kernel[1] = attr.kh;
  param.spec.common.kernel[2] = attr.kw;
  param.spec.common.stride[0] = attr.sd;
  param.spec.common.stride[1] = attr.sh;
  param.spec.common.stride[2] = attr.sw;
  param.spec.common.dilation[0] = attr.dd;
  param.spec.common.dilation[1] = attr.dh;
  param.spec.common.dilation[2] = attr.dw;
  param.spec.common.pad[0] = attr.pdf;
  param.spec.common.pad[1] = attr.pdb;
  param.spec.common.pad[2] = attr.pht;
  param.spec.common.pad[3] = attr.phb;
  param.spec.common.pad[4] = attr.pwl;
  param.spec.common.pad[5] = attr.pwr;
  param.spec.common.input_dtype = BM168x::getDataType(getInput());
  param.spec.common.weight_dtype = BM168x::getDataType(getFilter());
  param.spec.common.output_dtype = BM168x::getDataType(getOutput());
  param.spec.common.do_relu = attr.do_relu;
  param.spec.common.relu_limit = attr.relu_limit;
  if (module::getStorageType(getInput()).isIntOrIndex()) {
    auto out_etype = module::getStorageType(getOutput());
    param.spec.common.do_relu = out_etype.isUnsignedInteger();
    if (module::isUniformQuantized(getInput())) {
      auto in_qtype = module::getUniformQuantizedType(getInput());
      param.spec.common.pad_val = in_qtype.getZeroPoint();
    }
    param.spec.common.kzp_is_const = true;
    param.spec.common.kzp_val = attr.kernel_zp;
    param.spec.common.kzp_dtype = param.spec.common.weight_dtype;
    param.spec.common.pad_is_const = true;
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::Conv3DOp::get_fw_type_bm1684x() { return FW_BMNET_CONV3D; }

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Deconv.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

// ======================================
// GlobalGenInterface
// ======================================

void tpu::DeconvOp::codegen_global_bm1684x() {
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (getKernelShape().size() == 1) {
    BM168x::fix_shape(input_spec->at(0), {attr.n, attr.ic, attr.ih, attr.iw});
    BM168x::fix_shape(output_spec->at(0), {attr.n, attr.oc, attr.oh, attr.ow});
  }
  deconv_global_param_t param = {0};
  param.input_global_addr = module::getAddress(getInput());
  param.weight_global_addr = module::getAddress(getFilter());
  param.bias_global_addr = module::getAddress(getBias());
  param.output_global_addr = module::getAddress(getOutput());
  param.input_shape[0] = attr.n;
  param.input_shape[1] = attr.ic;
  param.input_shape[2] = attr.ih;
  param.input_shape[3] = attr.iw;
  param.groups = attr.g;
  param.output_c = attr.oc;
  param.kernel[0] = attr.kh;
  param.kernel[1] = attr.kw;
  param.stride[0] = attr.sh;
  param.stride[1] = attr.sw;
  param.dilation[0] = attr.dh;
  param.dilation[1] = attr.dw;
  param.pad[0] = attr.pad_h;
  param.pad[1] = attr.pad_h_after;
  param.pad[2] = attr.pad_w;
  param.pad[3] = attr.pad_w_after;
  param.output_pad[0] = attr.output_pad_h;
  param.output_pad[1] = attr.output_pad_w;
  param.has_bias = attr.with_bias;
  param.input_dtype = BM168x::getDataType(getInput());
  param.weight_dtype = BM168x::getDataType(getFilter());
  if (getBias().getType().isa<RankedTensorType>()) {
    param.bias_dtype = BM168x::getDataType(getBias());
  }
  param.output_dtype = BM168x::getDataType(getOutput());
  param.if_relu = attr.do_relu;
  param.upper_limit = attr.relu_limit;
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    param.is_asym = true;
    param.rshift = 0;
    param.kzp_global_addr = 0;
    param.pad_insert_global_addr = 0;
    param.kzp_is_const = true;
    param.pad_insert_is_const = true;
    param.kzp_val = 0;
    param.pad_val = in_qtype.getZeroPoint();
    param.insert_val = in_qtype.getZeroPoint();
    param.kzp_dtype = param.input_dtype;
  }
  BM168x::call_global_func("backend_api_deconv_global", &param, sizeof(param));
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::DeconvOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  int64_t sz = 0;
  auto &attr = getDeconvParam(*this);

  auto idtype = BM168x::getDataType(getInput());
  auto type_len = BM168x::getFmtBytes(idtype);
  int64_t eu_num = BM168x::eu_num(type_len);

  int ic_per_npu = ceiling_func(attr.ic / attr.g, BM168x::NPU_NUM);
  // fp part 2: used for group > 1, input must start from npu 0
  if (attr.g > 1 &&
      (idtype == DTYPE_FP32 || idtype == DTYPE_BFP16 || idtype == DTYPE_FP16)) {
    sz = ic_per_npu * align_up(in_hslice * in_wslice, eu_num) * type_len;
  }
  // quant : used for groups > 1, input must start from npu 0,
  if (attr.g > 1 && !attr.is_dw && type_len == 1) {
    sz = ic_per_npu * (align_up(in_hslice * in_wslice, eu_num) +
                       (attr.pad_insert_is_const ? 0 : 2));
  }

  return sz;
}

void tpu::DeconvOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                          int64_t h_step, int64_t d_step,
                                          int64_t w_step,
                                          group_type_t group_type,
                                          local_sec_info_t &sec_info) {
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (getKernelShape().size() == 1) {
    BM168x::fix_shape(input_spec->at(0), {attr.n, attr.ic, attr.ih, attr.iw});
    BM168x::fix_shape(output_spec->at(0), {attr.n, attr.oc, attr.oh, attr.ow});
  }
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step,
                                               d_step, w_step, c_step);
  auto filter_gi = LocalGenInterface::getGroupInfo(getFilter(), n_step, h_step,
                                                   d_step, w_step, c_step);
  auto bias_gi = LocalGenInterface::getGroupInfo(getBias(), n_step, h_step,
                                                 d_step, w_step, c_step);

  deconv_local_param_t param = {0};
  param.input_local_addr = (uint32_t)in_gi.out_addr;
  param.weight_local_addr = (uint32_t)filter_gi.out_addr;
  param.bias_local_addr = (uint32_t)bias_gi.out_addr;
  param.output_local_addr = (uint32_t)gi.out_addr;
  param.buffer_local_addr = gi.buffer_addr;
  param.input_shape[0] = sec_info.n_slice;
  param.input_shape[1] = attr.ic;
  param.input_shape[2] = sec_info.h_slice;
  param.input_shape[3] = sec_info.w_slice;
  param.groups = attr.g;
  param.output_c = attr.oc;
  param.kernel[0] = attr.kh;
  param.kernel[1] = attr.kw;
  param.stride[0] = attr.sh;
  param.stride[1] = attr.sw;
  param.dilation[0] = attr.dh;
  param.dilation[1] = attr.dw;
  int kh_ext = (attr.kh - 1) * attr.dh + 1;
  if (auto deconv_in_slice = DeconvSlice(gi.h_idx, gi.h_slice, attr.sh, kh_ext,
                                         attr.ih, attr.pad_h)) {
    param.pad[0] = deconv_in_slice.value()[0];
    param.pad[1] = deconv_in_slice.value()[1];
  } else {
    param.pad[0] = attr.kh - attr.pad_h - 1;
    param.pad[1] = attr.kh - attr.pad_h_after - 1;
  }
  int kw_ext = (attr.kw - 1) * attr.dw + 1;
  if (auto deconv_in_slice = DeconvSlice(gi.w_idx, gi.w_slice, attr.sw, kw_ext,
                                         attr.iw, attr.pad_w)) {
    param.pad[2] = deconv_in_slice.value()[0];
    param.pad[3] = deconv_in_slice.value()[1];
  } else {
    param.pad[2] = attr.kw - attr.pad_w - 1;
    param.pad[3] = attr.kw - attr.pad_w_after - 1;
  }
  param.has_bias = attr.with_bias;
  param.input_dtype = BM168x::getDataType(getInput());
  param.weight_dtype = BM168x::getDataType(getFilter());
  if (getBias().getType().isa<RankedTensorType>()) {
    param.bias_dtype = BM168x::getDataType(getBias());
  }

  param.output_dtype = BM168x::getDataType(getOutput());
  param.if_relu = attr.do_relu;
  param.upper_limit = attr.relu_limit;
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    param.is_asym = true;
    param.rshift = 0;
    param.kzp_local_addr = 0;
    param.pad_insert_local_addr = 0;
    param.kzp_is_const = true;
    param.pad_insert_is_const = true;
    param.kzp_val = 0;
    param.pad_val = in_qtype.getZeroPoint();
    param.insert_val = in_qtype.getZeroPoint();
    param.kzp_dtype = param.weight_dtype;
  }
  BM168x::call_local_func("backend_api_deconv_local", &param, sizeof(param));
}

// dynamic codegen
int64_t tpu::DeconvOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_deconv_local_spec_t);
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (getKernelShape().size() == 1) {
    BM168x::fix_shape(input_spec->at(0), {attr.n, attr.ic, attr.ih, attr.iw});
    BM168x::fix_shape(output_spec->at(0), {attr.n, attr.oc, attr.oh, attr.ow});
  }
  auto gi = getGroupInfo(0, 0, 0, 0, 0);

  dyn_deconv_local_spec_t param = {0};
  param.buffer_local_addr = gi.buffer_addr;
  param.common.groups = attr.g;
  param.common.output_c = attr.oc;
  param.common.kernel[0] = attr.kh;
  param.common.kernel[1] = attr.kw;
  param.common.stride[0] = attr.sh;
  param.common.stride[1] = attr.sw;
  param.common.dilation[0] = attr.dh;
  param.common.dilation[1] = attr.dw;
  param.common.pad[0] = attr.pad_h;
  param.common.pad[1] = attr.pad_h_after;
  param.common.pad[2] = attr.pad_w;
  param.common.pad[3] = attr.pad_w_after;
  param.common.output_pad[0] = attr.output_pad_h;
  param.common.output_pad[1] = attr.output_pad_w;
  param.common.has_bias = attr.with_bias;
  param.common.input_dtype = BM168x::getDataType(getInput());
  param.common.weight_dtype = BM168x::getDataType(getFilter());
  if (getBias().getType().isa<RankedTensorType>()) {
    param.common.bias_dtype = BM168x::getDataType(getBias());
  }

  param.common.output_dtype = BM168x::getDataType(getOutput());
  param.common.if_relu = attr.do_relu;
  param.common.upper_limit = attr.relu_limit;
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    param.common.is_asym = true;
    param.common.rshift = 0;
    param.kzp_local_addr = 0;
    param.pad_insert_local_addr = 0;
    param.common.kzp_is_const = true;
    param.common.pad_insert_is_const = true;
    param.common.kzp_val = 0;
    param.common.pad_val = in_qtype.getZeroPoint();
    param.common.insert_val = in_qtype.getZeroPoint();
    param.common.kzp_dtype = param.common.weight_dtype;
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::DeconvOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_deconv_global_spec_t);
  auto attr = parseParam();
  dyn_deconv_global_spec_t param = {0};
  param.common.groups = attr.g;
  param.common.output_c = attr.oc;
  param.common.kernel[0] = attr.kh;
  param.common.kernel[1] = attr.kw;
  param.common.stride[0] = attr.sh;
  param.common.stride[1] = attr.sw;
  param.common.dilation[0] = attr.dh;
  param.common.dilation[1] = attr.dw;
  param.common.pad[0] = attr.pad_h;
  param.common.pad[1] = attr.pad_h_after;
  param.common.pad[2] = attr.pad_w;
  param.common.pad[3] = attr.pad_w_after;
  param.common.output_pad[0] = param.output_pad[0] = attr.output_pad_h;
  param.common.output_pad[1] = param.output_pad[1] = attr.output_pad_w;
  param.common.has_bias = attr.with_bias;
  param.common.input_dtype = BM168x::getDataType(getInput());
  param.common.weight_dtype = BM168x::getDataType(getFilter());
  if (getBias().getType().isa<RankedTensorType>()) {
    param.common.bias_dtype = BM168x::getDataType(getBias());
  }
  param.common.output_dtype = BM168x::getDataType(getOutput());
  param.common.if_relu = attr.do_relu;
  param.common.upper_limit = attr.relu_limit;
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    param.common.is_asym = true;
    param.common.rshift = 0;
    param.kzp_global_addr = 0;
    param.pad_insert_global_addr = 0;
    param.common.kzp_is_const = true;
    param.common.pad_insert_is_const = true;
    param.common.kzp_val = 0;
    param.common.pad_val = in_qtype.getZeroPoint();
    param.common.insert_val = in_qtype.getZeroPoint();
    param.common.kzp_dtype = param.common.input_dtype;
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::DeconvOp::get_fw_type_bm1684x() { return FW_BMNET_DECONV; }

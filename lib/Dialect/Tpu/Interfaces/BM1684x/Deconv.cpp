//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Deconv.h"
#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

template <typename T>
static void filter_reorder(std::shared_ptr<std::vector<T>> &filter,
                           std::vector<int64_t> &shape) {
  int64_t oc, ic, kh, kw;
  Module::getNCHW(shape, oc, ic, kh, kw);
  auto type_bytes = sizeof(T);
  int64_t IC_PARALLEL = 64 / type_bytes;
  auto kernel_hw = kh * kw;
  int64_t new_ic = ceiling_func(ic, IC_PARALLEL);
  int64_t new_hw = kernel_hw * IC_PARALLEL;
  auto filter_new = std::make_shared<std::vector<T>>(oc * new_ic * new_hw, 0);
  for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < new_ic; ic_idx++) {
      for (int k_idx = 0; k_idx < kernel_hw; k_idx++) {
        for (int inner = 0; inner < IC_PARALLEL; inner++) {
          if (ic_idx * IC_PARALLEL + inner >= ic)
            break;
          int orig_offset = oc_idx * ic * kh * kw +
                            (ic_idx * IC_PARALLEL + inner) * kernel_hw + k_idx;
          int trans_offset = oc_idx * new_ic * new_hw + ic_idx * new_hw +
                             k_idx * IC_PARALLEL + inner;
          filter_new->at(trans_offset) = filter->at(orig_offset);
        }
      }
    }
  }
  filter = filter_new;
  shape = {1, oc, 1, new_ic * new_hw};
}

// refer to net_compiler: bool BM1684xCoeffArranger::DeconvWeightArr(GraphEdge*
// edge)
void tpu::DeconvOp::weight_reorder_int8_bm1684x() {
  // assume that ic = input_channel / groups, oc = output_channel / groups
  // for original model, deconv kernel is {groups * ic, oc, kh, kw},
  // but kernel is arranged to {groups * oc, ic, kh, kw} when adding_layer
  // here we arrange kernel to {groups * oc, ceil(ic, IC_PARALLEL), kh * kw * IC_PARALLEL}
  deconv_attr_t attrs;
  parseParam(&attrs);

  // filter op
  auto filterOp = filter().getDefiningOp<top::WeightOp>();
  auto filter_i8 = filterOp.read<int8_t>();
  auto filter_type = Module::getStorageType(filter());
  std::vector<int64_t> filter_shape = {attrs.oc, attrs.ic / attrs.g, attrs.kh, attrs.kw};
  int64_t IC_PARALLEL = 64;
  if (attrs.is_dw) {
    filter_shape = {1, attrs.oc, attrs.kh, attrs.kw};
    auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
    filter().setType(new_filter_type);
  } else {
    filter_reorder(filter_i8, filter_shape);
    auto op = getOperation();
    OpBuilder builder(getContext());
    auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
    auto newFilterOp = top::WeightOp::create(op, "_reordered", *filter_i8, new_filter_type);
    op->setOperand(1, newFilterOp);
  }

  // bias op
  if (attrs.with_bias) {
    auto biasOp = bias().getDefiningOp<top::WeightOp>();
    auto bias_type = Module::getStorageType(bias());
    int64_t bias_shape[4] = {1, attrs.oc, 1, 1};
    auto new_bias_type = RankedTensorType::get(bias_shape, bias_type);
    bias().setType(new_bias_type);
  }
}

void tpu::DeconvOp::weight_reorder_bf16_bm1684x() {
  deconv_attr_t attrs;
  parseParam(&attrs);

  // filter op
  auto filterOp = filter().getDefiningOp<top::WeightOp>();
  auto filter_i16 = filterOp.read<int16_t>();
  auto filter_type = Module::getStorageType(filter());
  std::vector<int64_t> filter_shape = {attrs.oc, attrs.ic / attrs.g, attrs.kh, attrs.kw};
  int64_t IC_PARALLEL = 32;
  if (attrs.is_dw) {
    filter_shape = {1, attrs.oc, attrs.kh, attrs.kw};
    auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
    filter().setType(new_filter_type);
  } else {
    filter_reorder(filter_i16, filter_shape);
    auto op = getOperation();
    OpBuilder builder(getContext());
    auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
    auto newFilterOp = top::WeightOp::create(op, "_reordered", *filter_i16, new_filter_type);
    op->setOperand(1, newFilterOp);
  }

  // bias op
  if (attrs.with_bias) {
    auto biasOp = bias().getDefiningOp<top::WeightOp>();
    auto bias_type = Module::getStorageType(bias());
    int64_t bias_shape[4] = {1, attrs.oc, 1, 1};
    auto new_bias_type = RankedTensorType::get(bias_shape, bias_type);
    bias().setType(new_bias_type);
  }
}

void tpu::DeconvOp::weight_reorder_f16_bm1684x() {
  weight_reorder_bf16_bm1684x();
}

void tpu::DeconvOp::weight_reorder_f32_bm1684x() {
  deconv_attr_t attrs;
  parseParam(&attrs);

  // filter op
  auto filterOp = filter().getDefiningOp<top::WeightOp>();
  auto filter_type = Module::getStorageType(filter());
  std::vector<int64_t> filter_shape = {1, attrs.oc, attrs.ic / attrs.g, attrs.kh * attrs.kw};
  auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
  filter().setType(new_filter_type);

  // bias op
  if (attrs.with_bias) {
    auto biasOp = bias().getDefiningOp<top::WeightOp>();
    auto bias_type = Module::getStorageType(bias());
    int64_t bias_shape[4] = {1, attrs.oc, 1, 1};
    auto new_bias_type = RankedTensorType::get(bias_shape, bias_type);
    bias().setType(new_bias_type);
  }
}

// ======================================
// GlobalGenInterface
// ======================================

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  /* common param */
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long output_global_addr;
  int input_shape[4];
  int groups;
  int output_c;
  int kernel[2];     // (kh, kw)
  int stride[2];     // (h, w)
  int dilation[2];   // (h, w)
  int pad[4];        // (h0, h1, w0, w1)
  int output_pad[2]; // (h, w)
  int has_bias;
  int input_dtype;
  int weight_dtype;
  int bias_dtype;
  /* param for float */
  int output_dtype;
  int if_relu;
  float upper_limit;
  /* param for quant */
  bool is_asym;
  unsigned char rshift;
  unsigned long long kzp_global_addr;
  unsigned long long pad_insert_global_addr;
  bool kzp_is_const;
  bool pad_insert_is_const;
  int kzp_val;
  int pad_val;
  int insert_val;
  int kzp_dtype;
} deconv_global_param_t;

typedef struct {
  /* common param */
  unsigned int input_local_addr;
  unsigned int weight_local_addr;
  unsigned int bias_local_addr;
  unsigned int buffer_local_addr;
  unsigned int output_local_addr;
  int input_shape[4];
  int groups;
  int output_c;
  int kernel[2];   // (kh, kw)
  int stride[2];   // (h, w)
  int dilation[2]; // (h, w)
  int pad[4];      // (h0, h1, w0, w1)
  int has_bias;
  int input_dtype;
  int weight_dtype;
  int bias_dtype;
  /* param for float */
  int output_dtype;
  int if_relu;
  float upper_limit;
  /* param for quant */
  bool is_asym;
  unsigned char rshift;
  unsigned int kzp_local_addr;
  unsigned int pad_insert_local_addr;
  bool kzp_is_const;
  bool pad_insert_is_const;
  int kzp_val;
  int pad_val;
  int insert_val;
  int kzp_dtype;
} deconv_local_param_t;

#ifdef __cplusplus
}
#endif

void tpu::DeconvOp::codegen_global_int8_bm1684x() {
  deconv_attr_t attrs;
  parseParam(&attrs);

  deconv_global_param_t param = {0};
  param.input_global_addr = Module::getAddress(input());
  param.weight_global_addr = Module::getAddress(filter());
  param.bias_global_addr = Module::getAddress(bias());
  param.output_global_addr = Module::getAddress(output());
  param.input_shape[0] = attrs.n;
  param.input_shape[1] = attrs.ic;
  param.input_shape[2] = attrs.ih;
  param.input_shape[3] = attrs.iw;
  param.groups = attrs.g;
  param.output_c = attrs.oc;
  param.kernel[0] = attrs.kh;
  param.kernel[1] = attrs.kw;
  param.stride[0] = attrs.sh;
  param.stride[1] = attrs.sw;
  param.dilation[0] = attrs.dh;
  param.dilation[1] = attrs.dw;
  param.pad[0] = attrs.pad_h;
  param.pad[1] = attrs.pad_h_after;
  param.pad[2] = attrs.pad_w;
  param.pad[3] = attrs.pad_w_after;
  param.output_pad[0] = attrs.output_pad_h;
  param.output_pad[1] = attrs.output_pad_w;
  param.has_bias = attrs.with_bias;
  param.input_dtype = BM1684x::getDataType(input());
  param.weight_dtype = BM1684x::getDataType(filter());
  if (bias().getType().isa<RankedTensorType>()) {
    param.bias_dtype = BM1684x::getDataType(bias());
  }

  param.output_dtype = BM1684x::getDataType(output());
  param.if_relu = attrs.do_relu;
  param.upper_limit = attrs.relu_limit;

  auto in_qtype = Quant::getUniformQuantizedType(input());
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
  BM1684x::instance().call_global_func("backend_api_deconv_global", &param,
                                       sizeof(param));
}

// f32
void tpu::DeconvOp::codegen_global_float_bm1684x() {
  deconv_attr_t attrs;
  parseParam(&attrs);

  deconv_global_param_t param = {0};
  param.input_global_addr = Module::getAddress(input());
  param.weight_global_addr = Module::getAddress(filter());
  param.bias_global_addr = Module::getAddress(bias());
  param.output_global_addr = Module::getAddress(output());
  param.input_shape[0] = attrs.n;
  param.input_shape[1] = attrs.ic;
  param.input_shape[2] = attrs.ih;
  param.input_shape[3] = attrs.iw;
  param.groups = attrs.g;
  param.output_c = attrs.oc;
  param.kernel[0] = attrs.kh;
  param.kernel[1] = attrs.kw;
  param.stride[0] = attrs.sh;
  param.stride[1] = attrs.sw;
  param.dilation[0] = attrs.dh;
  param.dilation[1] = attrs.dw;
  param.pad[0] = attrs.pad_h;
  param.pad[1] = attrs.pad_h_after;
  param.pad[2] = attrs.pad_w;
  param.pad[3] = attrs.pad_w_after;
  param.output_pad[0] = attrs.output_pad_h;
  param.output_pad[1] = attrs.output_pad_w;
  param.has_bias = attrs.with_bias;
  param.input_dtype = BM1684x::getDataType(input());
  param.weight_dtype = BM1684x::getDataType(filter());
  if (bias().getType().isa<RankedTensorType>()) {
    param.bias_dtype = BM1684x::getDataType(bias());
  }

  param.output_dtype = BM1684x::getDataType(output());
  param.if_relu = attrs.do_relu;
  param.upper_limit = attrs.relu_limit;
  BM1684x::instance().call_global_func("backend_api_deconv_global", &param,
                                       sizeof(param));
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::DeconvOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t sz = out_lmem_bytes * sizeof(int32_t);
  deconv_attr_t attrs;
  parseParam(&attrs);

  auto idtype = BM168x::getDataType(input());
  int type_len = BM168x::getFmtBytes(idtype);
  int64_t eu_num = BM1684x::instance().get_eu_num(type_len);
  int ic_per_npu = ceiling_func(attrs.ic / attrs.g, BM1684x::NPU_NUM);
  // fp part 2: used for group > 1, input must start from npu 0
  if (attrs.g > 1 &&
      (idtype == DTYPE_FP32 || idtype == DTYPE_BFP16 || idtype == DTYPE_FP16)) {
    sz += ic_per_npu * align_up(in_hslice * attrs.iw, eu_num) * type_len;
  }
  // quant : used for groups > 1, input must start from npu 0,
  if (attrs.g > 1 && !attrs.is_dw && type_len == 1) {
    sz += ic_per_npu *
          (align_up(in_hslice * attrs.iw, eu_num) + attrs.pad_insert_is_const ? 0
                                                                           : 2);
  }

  return sz;
}

void tpu::DeconvOp::codegen_local_int8_bm1684x(int64_t n_step, int64_t h_step) {
  deconv_attr_t attrs;
  parseParam(&attrs);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto filter_gi = LocalGenInterface::getGroupInfo(filter(), n_step, h_step);
  auto bias_gi = LocalGenInterface::getGroupInfo(bias(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);

  deconv_local_param_t param = {0};
  param.input_local_addr = (uint32_t)in_gi.out_addr;
  param.weight_local_addr = (uint32_t)filter_gi.out_addr;
  param.bias_local_addr = (uint32_t)bias_gi.out_addr;
  param.output_local_addr = (uint32_t)gi.out_addr;
  param.buffer_local_addr = gi.buffer_addr;
  param.input_shape[0] = in_gi.n_slice;
  param.input_shape[1] = attrs.ic;
  param.input_shape[2] = in_gi.h_slice;
  param.input_shape[3] = attrs.iw;
  param.groups = attrs.g;
  param.output_c = attrs.oc;
  param.kernel[0] = attrs.kh;
  param.kernel[1] = attrs.kw;
  param.stride[0] = attrs.sh;
  param.stride[1] = attrs.sw;
  param.dilation[0] = attrs.dh;
  param.dilation[1] = attrs.dw;
  param.pad[0] = attrs.pad_h;
  param.pad[1] = attrs.pad_h_after;
  param.pad[2] = attrs.pad_w;
  param.pad[3] = attrs.pad_w_after;
  param.has_bias = attrs.with_bias;
  param.input_dtype = BM1684x::getDataType(input());
  param.weight_dtype = BM1684x::getDataType(filter());
  if (bias().getType().isa<RankedTensorType>()) {
    param.bias_dtype = BM1684x::getDataType(bias());
  }

  param.output_dtype = BM1684x::getDataType(output());
  param.if_relu = attrs.do_relu;
  param.upper_limit = attrs.relu_limit;

  auto in_qtype = Quant::getUniformQuantizedType(input());
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
  BM1684x::instance().call_local_func("backend_api_deconv_local", &param,
                                      sizeof(param));
}

void tpu::DeconvOp::codegen_local_float_bm1684x(int64_t n_step,
                                                int64_t h_step) {
  deconv_attr_t attrs;
  parseParam(&attrs);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto filter_gi = LocalGenInterface::getGroupInfo(filter(), n_step, h_step);
  auto bias_gi = LocalGenInterface::getGroupInfo(bias(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);

  deconv_local_param_t param = {0};
  param.input_local_addr = (uint32_t)in_gi.out_addr;
  param.weight_local_addr = (uint32_t)filter_gi.out_addr;
  param.bias_local_addr = (uint32_t)bias_gi.out_addr;
  param.output_local_addr = (uint32_t)gi.out_addr;
  param.buffer_local_addr = gi.buffer_addr;
  param.input_shape[0] = in_gi.n_slice;
  param.input_shape[1] = attrs.ic;
  param.input_shape[2] = in_gi.h_slice;
  param.input_shape[3] = attrs.iw;
  param.groups = attrs.g;
  param.output_c = attrs.oc;
  param.kernel[0] = attrs.kh;
  param.kernel[1] = attrs.kw;
  param.stride[0] = attrs.sh;
  param.stride[1] = attrs.sw;
  param.dilation[0] = attrs.dh;
  param.dilation[1] = attrs.dw;
  param.pad[0] = attrs.pad_h;
  param.pad[1] = attrs.pad_h_after;
  param.pad[2] = attrs.pad_w;
  param.pad[3] = attrs.pad_w_after;
  param.has_bias = attrs.with_bias;
  param.input_dtype = BM1684x::getDataType(input());
  param.weight_dtype = BM1684x::getDataType(filter());
  if (bias().getType().isa<RankedTensorType>()) {
    param.bias_dtype = BM1684x::getDataType(bias());
  }

  param.output_dtype = BM1684x::getDataType(output());
  param.if_relu = attrs.do_relu;
  param.upper_limit = attrs.relu_limit;
  BM1684x::instance().call_local_func("backend_api_deconv_local", &param,
                                      sizeof(param));
}

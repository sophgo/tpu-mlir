//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Dnnl/Conv.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::Conv2DOp::codegen_global_bm1684() {
  auto attr = parseParam();
  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  auto filter_addr = module::getAddress(getFilter());
  auto bias_addr = module::getAddress(getBias());
  if (module::isUniformQuantized(getInput())) {
    auto shift_v = module::getI64Array(getRshift(), 1, 0);
    auto shift = shift_v->at(0);
    auto in_sign = module::isSign(getInput());
    auto filter_sign = module::isSign(getFilter());
    auto bias_sign = attr.has_bias ? module::isSign(getBias()) : 0;
    auto out_sign = module::isSign(getOutput());
    if (attr.is_dw) {
      BM1684::instance().dl_nodechip_depthwise_fix8b_forward_parallel(
          in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic, attr.ih,
          attr.iw, attr.kh, attr.kw, attr.pht, attr.phb, attr.pwl, attr.pwr,
          attr.sh, attr.sw, attr.ins_h, attr.ins_w, shift,
          attr.has_bias ? 1 : 0, /*shift_sign*/ 0, in_sign, filter_sign,
          bias_sign, out_sign, attr.do_relu ? 1 : 0, attr.relu_limit,
          (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    } else {
      BM1684::instance()
          .dl_nodechip_conv_forward_parallel_fix8b_with_data_split(
              in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic,
              attr.ih, attr.iw, attr.groups, attr.oc, attr.kh, attr.kw, attr.dh,
              attr.dw, attr.pht, attr.phb, attr.pwl, attr.pwr, attr.sh, attr.sw,
              attr.has_bias ? 1 : 0, 0, attr.do_relu ? 1 : 0, 0, 1, 0, 0, shift,
              in_sign, filter_sign, bias_sign, 3, 0, 0, 0, 0, 0,
              (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    }
  } else {
    // F32
    if (attr.is_dw) {
      BM1684::instance().dl_nodechip_depthwise_forward_parallel(
          in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic, attr.ih,
          attr.iw, attr.kh, attr.kw, attr.pht, attr.phb, attr.pwl, attr.pwr,
          attr.sh, attr.sw, attr.dh, attr.dw, attr.has_bias ? 1 : 0,
          attr.do_relu ? 1 : 0, attr.relu_limit,
          (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    } else {
      BM1684::instance().dl_nodechip_conv_forward_parallel_with_data_split(
          in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic, attr.ih,
          attr.iw, attr.groups, attr.oc, attr.kh, attr.kw, attr.dh, attr.dw,
          attr.pht, attr.phb, attr.pwl, attr.pwr, attr.sh, attr.sw,
          attr.has_bias ? 1 : 0, 0 /* result_add*/, attr.do_relu ? 1 : 0,
          attr.relu_limit, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    }
  }
}

int64_t tpu::Conv2DOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::Conv2DOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto f_gi = LocalGenInterface::getGroupInfo(getFilter());
  auto b_gi = LocalGenInterface::getGroupInfo(getBias());
  auto gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto p = parseParam();
  int bottom_dim[4] = {(int)in_gi.n_slice, (int)p.ic, (int)in_gi.h_slice,
                       (int)p.iw};
  int top_dim[4] = {(int)gi.n_slice, (int)p.oc, (int)gi.h_slice, (int)p.ow};
  auto pad_h_t = (in_gi.h_idx == 0 ? p.pht : 0);
  auto pad_h_b = (in_gi.h_idx + in_gi.h_slice == p.ih ? p.phb : 0);
  if (module::isUniformQuantized(getInput())) {
    auto shift_v = module::getI64Array(getRshift(), 1, 0);
    auto shift = shift_v->at(0);
    auto in_sign = module::isSign(getInput());
    auto filter_sign = module::isSign(getFilter());
    auto bias_sign = p.has_bias ? module::isSign(getBias()) : 0;
    auto out_sign = module::isSign(getOutput());
    if (p.is_dw) {
      BM1684::instance().dl_nodechip_pooling_fix8b_forward_local(
          in_gi.out_addr, f_gi.out_addr, b_gi.out_addr, gi.out_addr, bottom_dim,
          top_dim, p.kh, p.kw, pad_h_t, pad_h_b, p.pwl, p.pwr, p.sh, p.sw,
          p.ins_h, p.ins_w,
          2, // is depthwise conv
          0, shift, p.has_bias,
          1, // shift type, useless param, but must be set...
          in_sign, filter_sign, bias_sign, out_sign, p.do_relu,
          BM1684::instance().bdc_node);
    } else {
      BM1684::instance().dl_nodechip_conv_forward_local_fix8b(
          in_gi.out_addr, f_gi.out_addr, b_gi.out_addr, gi.out_addr,
          gi.buffer_addr, bottom_dim, top_dim, p.groups, p.kh, p.kw, p.dh, p.dw,
          pad_h_t, pad_h_b, p.pwl, p.pwr, p.sh, p.sw, p.has_bias, 0, p.do_relu,
          p.relu_limit, /*unused_ht*/ 0, 0, 0, 0, /* insert h*/ p.ins_h,
          p.ins_w, shift, in_sign, filter_sign, bias_sign, true, /*mulshift*/ 0,
          0, 0, 0, BM1684::instance().bdc_node);
    }
  } else {
    BM1684::instance().dl_nodechip_conv_forward_local(
        in_gi.out_addr, f_gi.out_addr, b_gi.out_addr, gi.out_addr,
        gi.buffer_addr, bottom_dim, top_dim, p.groups, p.kh, p.kw, p.dh, p.dw,
        pad_h_t, pad_h_b, p.pwl, p.pwr, p.sh, p.sw, p.has_bias ? 1 : 0,
        /* result_add*/ 0, p.do_relu ? 1 : 0, p.relu_limit, 0, 0, 0, 0,
        BM1684::instance().bdc_node);
  }
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================

uint32_t tpu::Conv2DOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  uint32_t fw_ir_length = 0;
  ir_layer_info_t *conv_layer_info = (ir_layer_info_t *)ir_layer_info;
  conv_layer_info->data_size =
      get_dynamic_compiler_tensor_datasize(getOutput());
  conv_layer_info->intensor_store_mode = BM168x::getStoreMode(getInput());
  conv_layer_info->outtensor_store_mode = BM168x::getStoreMode(getOutput());

  fw_conv_layer_param_t fw_conv_layer_param = {0};
  assign_fw_param((void *)&fw_conv_layer_param);

  conv_layer_info->fw_layer_param_u.fw_conv_layer_param = fw_conv_layer_param;
  fw_ir_length += sizeof(fw_conv_layer_param_t);

  return fw_ir_length;
}

int64_t tpu::Conv2DOp::get_fw_type_bm1684() { return FW_BMNET_CONV; }

// ======================================
// Dynamic LocalGenInterface
// ======================================

int32_t tpu::Conv2DOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  int fw_ir_length = 0;
  ir_layer_info_t *conv_layer_info = (ir_layer_info_t *)ir_layer_info;
  conv_layer_info->data_size =
      get_dynamic_compiler_tensor_datasize(getOutput());
  conv_layer_info->intensor_store_mode = BM168x::getStoreMode(getInput());
  conv_layer_info->outtensor_store_mode = BM168x::getStoreMode(getOutput());

  // get fw layer param
  fw_conv_layer_param_t fw_conv_layer_param = {0};
  assign_fw_param((void *)&fw_conv_layer_param);

  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), 0, 0);
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  fw_conv_layer_param.c_idx = 0; // not support c_slice now
  fw_conv_layer_param.reference_id = get_tensor_id(getOutput());
  fw_conv_layer_param.concat_c = c;

  fw_conv_layer_param.if_double_buffer = 0;
  int using_immbuf = 0; // ref to Conv2DOp::getBufferSize_bm1684
  fw_conv_layer_param.if_relu |=
      (using_immbuf << 4); // conv's if_relu high 4 bit used as immbuf flag

  uint64_t weight_global_addr = module::getAddress(getFilter());
  fw_conv_layer_param.weight_global_offset = weight_global_addr;
  fw_conv_layer_param.double_buffer_local_offset = 0;

  conv_layer_info->fw_layer_param_u.fw_conv_layer_param = fw_conv_layer_param;
  fw_ir_length += sizeof(fw_conv_layer_param_t);

  // get layer input and output
  conv_layer_info->ir_tensor_info_v.clear();
  // input tensor
  dynamic_push_back_local_tensor(conv_layer_info->ir_tensor_info_v, getInput());

  // weight
  dynamic_push_back_local_tensor(conv_layer_info->ir_tensor_info_v,
                                 getFilter());

  // bias
  if (getWithBias()) {
    dynamic_push_back_local_tensor(conv_layer_info->ir_tensor_info_v,
                                   getBias());
  }

  // output, in local concat case, let the out_tensor_id which is the first conv
  // of concat be concat's out_tensor_id
  dynamic_push_back_local_tensor(conv_layer_info->ir_tensor_info_v,
                                 getOutput());

  // compute fw ir info length for conv input and output
  fw_ir_length += (sizeof(uint32_t) + 3 * sizeof(uint32_t) +
                   (getWithBias() ? 1 : 0) * sizeof(uint32_t));

  // add fw ir length for output consumer number
  fw_ir_length += sizeof(uint32_t);

  return fw_ir_length;
}

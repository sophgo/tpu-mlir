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
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

// static int inline get_tensor_local_size(const int *shape, int type_len,
//                                         STORE_MODE_T store_mode, int npu_num,
//                                         int eu_num, bool is_aligned = true) {
//   int n = shape[0];
//   if (store_mode == STORE_MODE_4N && type_len == 1) {
//     n = align_up(n, 4);
//   } else if (store_mode == STORE_MODE_2N && type_len == 2) {
//     n = align_up(n, 2);
//   }
//   int c = shape[1];
//   int h = shape[2];
//   int w = shape[3];
//   int cnum = (c + eu_num - 1) / npu_num;
//   int cstride = h * w;
//   if (is_aligned) {
//     cstride = align_up(cstride, (eu_num * 4 / type_len));
//   }
//   int buffer_size = n * cnum * cstride * type_len;
//   return buffer_size;
// }

void tpu::ConvBwdWeightOp::codegen_global_bm1684() {
  // auto attr = parseParam();
  // auto in_addr = module::getAddress(getInput());
  // auto out_addr = module::getAddress(getOutput());
  // auto filter_addr = module::getAddress(getFilter());
  // auto bias_addr = module::getAddress(getBias());
  // if (module::isUniformQuantized(getInput())) {
  //   auto shift_v =
  //       module::getI64Array(getRshift(), attr.use_winograd == 0 ? 1 : 2, 0);
  //   auto out_etype = module::getStorageType(getOutput());
  //   if (out_etype.isUnsignedInteger()) {
  //     attr.do_relu = true;
  //   }
  //   auto shift = shift_v->at(0);
  //   auto in_sign = module::isSign(getInput());
  //   auto filter_sign = module::isSign(getFilter());
  //   auto bias_sign = attr.has_bias ? module::isSign(getBias()) : 0;
  //   auto out_sign = module::isSign(getOutput());
  //   if (attr.is_dw) {
  //     BM1684::instance().dl_nodechip_depthwise_fix8b_forward_parallel(
  //         in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic,
  //         attr.ih, attr.iw, attr.kh, attr.kw, attr.pht, attr.phb, attr.pwl,
  //         attr.pwr, attr.sh, attr.sw, attr.ins_h, attr.ins_w, shift,
  //         attr.has_bias ? 1 : 0, /*shift_sign*/ 0, in_sign, filter_sign,
  //         bias_sign, out_sign, attr.do_relu ? 1 : 0, attr.relu_limit,
  //         (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  //   } else {
  //     if (attr.use_winograd == 2) {
  //       //
  //       /workspace/nntoolchain/net_compiler/bmcompiler/src/cmd_gen/bm1684_global_layer_ctrl.cpp
  //       // dl_nodechip_winograd_forward_parallel_with_data_split
  //       int64_t bias_global_offset =
  //           filter_addr + ceiling_func(attr.oc / attr.groups, 64) *
  //                             ceiling_func(attr.ic / attr.groups, 4) * 64;

  //       BM1684::instance()
  //           .dl_nodechip_winograd_forward_parallel_fix8b_with_data_split(
  //               in_addr, out_addr, filter_addr, bias_global_offset, attr.n,
  //               attr.ic, attr.ih, attr.iw, attr.groups, attr.oc, attr.pht,
  //               attr.phb, attr.pwl, attr.pwr, 2, attr.do_relu ? 1 : 0,
  //               attr.relu_limit, 2, 0, 0, shift_v->at(1), in_sign,
  //               filter_sign, bias_sign, 3, 0, 0, 0, 0, (CMD_ID_NODE
  //               *)BM1684::instance()->cmdid_node);
  //     } else {
  //       // int bottom_height = attr.ih;
  //       // if (getUse_3icOptimize())
  //       //   bottom_height += attr.pht + attr.phb;
  //       BM1684::instance()
  //           .dl_nodechip_conv_forward_parallel_fix8b_with_data_split(
  //               in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic,
  //               attr.ih, attr.iw, attr.groups, attr.oc, attr.kh, attr.kw,
  //               attr.dh, attr.dw, attr.pht, attr.phb, attr.pwl, attr.pwr,
  //               attr.sh, attr.sw, attr.has_bias ? 1 : 0, 0,
  //               attr.do_relu ? 1 : 0, 0, 1, 0, 0, shift, in_sign,
  //               filter_sign, bias_sign, 3, 0, 0, 0, 0, getUse_3icOptimize(),
  //               (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  //     }
  //   }
  // } else {
  //   // F32
  //   if (attr.is_dw) {
  //     BM1684::instance().dl_nodechip_depthwise_forward_parallel(
  //         in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic,
  //         attr.ih, attr.iw, attr.kh, attr.kw, attr.pht, attr.phb, attr.pwl,
  //         attr.pwr, attr.sh, attr.sw, attr.dh, attr.dw, attr.has_bias ? 1 :
  //         0, attr.do_relu ? 1 : 0, attr.relu_limit, (CMD_ID_NODE
  //         *)BM1684::instance()->cmdid_node);
  //   } else {
  //     BM1684::instance().dl_nodechip_conv_forward_parallel_with_data_split(
  //         in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic,
  //         attr.ih, attr.iw, attr.groups, attr.oc, attr.kh, attr.kw, attr.dh,
  //         attr.dw, attr.pht, attr.phb, attr.pwl, attr.pwr, attr.sh, attr.sw,
  //         attr.has_bias ? 1 : 0, 0 /* result_add*/, attr.do_relu ? 1 : 0,
  //         attr.relu_limit, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  //   }
  // }
}

int64_t tpu::ConvBwdWeightOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  // auto p = parseParam();
  // auto pads_v = module::getI64Array(getPads());
  // p.pht = pads_v->at(0);
  // p.pwl = pads_v->at(1);
  // p.phb = pads_v->at(2);
  // p.pwr = pads_v->at(3);
  // int buffer_size = 0;
  // auto dtype_i = BM168x::getDataType(getInput());
  // auto NPU_NUM = BM168x::NPU_NUM;
  // auto EU_NUM = BM168x::eu_num(sizeof(float));
  // bool need_buffer = p.is_dw;
  // if (dtype_i == DTYPE_FP32) {
  //   if (p.is_dw == 0 &&
  //       (in_hslice + p.pht + p.phb > 2047 || p.iw + p.pwl + p.pwr > 2047)) {
  //     need_buffer = true;
  //   }
  //   if (need_buffer) {
  //     int h_pad = in_hslice + p.pht + p.phb;
  //     int w_pad = p.iw + p.pwl + p.pwr;
  //     int real_h =
  //         ((h_pad > 2047 ? 2047 : h_pad) - ((p.kh - 1) * p.dh + 1)) / p.sh +
  //         1;
  //     int real_w =
  //         ((w_pad > 2047 ? 2047 : w_pad) - ((p.kw - 1) * p.dw + 1)) / p.sw +
  //         1;
  //     // split_shape[0][0] = n_slice;
  //     int new_shape[4] = {(int)in_nslice, (int)p.oc, real_h, real_w};
  //     buffer_size = get_tensor_local_size(new_shape, 4, STORE_MODE_1N,
  //     (int)NPU_NUM, (int)EU_NUM, true);
  //   }
  //   return buffer_size;
  // } else {
  //   if (dtype_i == DTYPE_INT16 || dtype_i == DTYPE_UINT16) {
  //     buffer_size = ceiling_func(in_nslice, 4) * ceiling_func(p.ic, NPU_NUM)
  //     *
  //                   align_up((in_hslice + p.pht + p.phb) * p.iw, EU_NUM) *
  //                   sizeof(int);
  //     int kh_ext = p.dh * (p.kh - 1) + 1;
  //     int output_h = (in_hslice + p.pht + p.phb - kh_ext) / p.sh + 1;
  //     // for 4N int16 output data
  //     buffer_size += ceiling_func(in_nslice, 2) * output_h *
  //                    align_up(1 * 32, EU_NUM) * 2 * sizeof(int);
  //     return buffer_size;
  //   }
  //   //
  //   workd/nntoolchain/net_compiler/bmcompiler/src/layers/bm1684/bm1684_conv.cpp
  //   // ##todo int8/uint8 datatype only when merged mulshift into conv need
  //   buffer;
  // }
  return 0;
}

void tpu::ConvBwdWeightOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                                local_sec_info_t &sec_info) {
  // auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  // auto f_gi = LocalGenInterface::getGroupInfo(getFilter());
  // auto b_gi = LocalGenInterface::getGroupInfo(getBias());
  // auto gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  // auto p = parseParam();
  // int bottom_dim[4] = {(int)in_gi.n_slice, (int)p.ic, (int)in_gi.h_slice,
  //                      (int)p.iw};
  // int top_dim[4] = {(int)gi.n_slice, (int)p.oc, (int)gi.h_slice, (int)p.ow};
  // auto pad_h_t = (in_gi.h_idx == 0 ? p.pht : 0);
  // auto pad_h_b = (in_gi.h_idx + in_gi.h_slice == p.ih ? p.phb : 0);

  // int unused_ht_for_input = 0, unused_hb_for_input = 0, unused_wl_for_input =
  // 0,
  //     unused_wr_for_input = 0;
  // int64_t N, C, H, W;
  // module::getNCHW(getInput(), N, C, H, W);
  // if (sec_info.h_slice != H) {
  //   int use_winograd = getUseWinograd().value_or(0);
  //   int kh_consider_dh = use_winograd > 0 ? 3 : (p.kh - 1) * (p.dh) + 1;
  //   int cal_h_idx = sec_info.out_h_idx * p.sh - p.pht;
  //   int cal_h_slice = (sec_info.out_h_slice - 1) * p.sh + kh_consider_dh;
  //   cal_h_slice = std::min(cal_h_slice, cal_h_slice + cal_h_idx);
  //   cal_h_idx = std::max(0, cal_h_idx);
  //   unused_ht_for_input = cal_h_idx - std::max(0, sec_info.h_idx);
  //   int h_end = std::min(sec_info.h_idx + sec_info.h_slice, (int)H);
  //   unused_hb_for_input = std::max(0, h_end - (cal_h_idx + cal_h_slice));
  // }

  // if (sec_info.w_slice != W) {
  //   int use_winograd = getUseWinograd().value_or(0);
  //   int kw_consider_dw = use_winograd > 0 ? 3 : (p.kw - 1) * (p.dw) + 1;
  //   int cal_w_idx = sec_info.out_w_idx * p.sw - p.pwl;
  //   int cal_w_slice = (sec_info.out_w_slice - 1) * p.sw + kw_consider_dw;
  //   cal_w_slice = std::min(cal_w_slice, cal_w_slice + cal_w_idx);
  //   cal_w_idx = std::max(0, cal_w_idx);

  //   unused_wl_for_input = cal_w_idx - std::max(0, sec_info.w_idx);
  //   int w_end = std::min(sec_info.w_idx + sec_info.w_slice, (int)W);
  //   unused_wr_for_input = std::max(0, w_end - (cal_w_idx + cal_w_slice));
  // }

  // if (module::isUniformQuantized(getInput())) {
  //   int use_winograd = getUseWinograd().value_or(0);
  //   auto shift_v = module::getI64Array(getRshift(), use_winograd ? 2 : 1, 0);
  //   auto shift = shift_v->at(0);
  //   auto in_sign = module::isSign(getInput());
  //   auto filter_sign = module::isSign(getFilter());
  //   auto bias_sign = p.has_bias ? module::isSign(getBias()) : 0;
  //   auto out_sign = module::isSign(getOutput());
  //   if (use_winograd == 2) {
  //     auto winorshift = shift_v->at(1);
  //     auto bias_local_addr =
  //         f_gi.out_addr + align_up(p.oc / p.groups, 4) *
  //                             ceiling_func(p.ic / p.groups, 64) * 4 * 4;
  //     BM1684::instance().dl_nodechip_winograd_forward_local_fix8b(
  //         in_gi.out_addr, f_gi.out_addr, bias_local_addr, gi.out_addr,
  //         gi.buffer_addr, bottom_dim, top_dim, p.groups, pad_h_t, pad_h_b,
  //         p.pwl, p.pwr, 2 /* use_bias*/, 0, p.do_relu, p.relu_limit, 2,
  //         /*unused_ht*/ 0, 0, 0, 0, winorshift, in_sign, filter_sign,
  //         bias_sign, 3,
  //         /*mulshift*/ 0, 0, 0, 0, BM1684::instance()->bdc_node);
  //   } else if (p.is_dw) {
  //     BM1684::instance().dl_nodechip_pooling_fix8b_forward_local(
  //         in_gi.out_addr, f_gi.out_addr, b_gi.out_addr, gi.out_addr,
  //         bottom_dim, top_dim, p.kh, p.kw, pad_h_t, pad_h_b, p.pwl, p.pwr,
  //         p.sh, p.sw, p.ins_h, p.ins_w, 2, // is depthwise conv 0, shift,
  //         p.has_bias, 1, // shift type, useless param, but must be set...
  //         in_sign, filter_sign, bias_sign, out_sign, p.do_relu,
  //         BM1684::instance()->bdc_node);
  //   } else {
  //     bottom_dim[1] =
  //         getUse_3icOptimize() ? bottom_dim[1] * p.kh : bottom_dim[1];
  //     bottom_dim[2] = getUse_3icOptimize() ? top_dim[2] : bottom_dim[2];
  //     // pad_h_t = getUse_3icOptimize() ? 0 : pad_h_t;
  //     // pad_h_b = getUse_3icOptimize() ? 0 : pad_h_b;
  //     BM1684::instance().dl_nodechip_conv_forward_local_fix8b(
  //         in_gi.out_addr, f_gi.out_addr, b_gi.out_addr, gi.out_addr,
  //         gi.buffer_addr, bottom_dim, top_dim, p.groups,
  //         getUse_3icOptimize() ? 1 : p.kh, p.kw, p.dh, p.dw, pad_h_t,
  //         pad_h_b, p.pwl, p.pwr, getUse_3icOptimize() ? 1 : p.sh, p.sw,
  //         p.has_bias, 0, p.do_relu, p.relu_limit, /*unused_ht*/
  //         unused_ht_for_input, unused_hb_for_input, unused_wl_for_input,
  //         unused_wr_for_input,
  //         /* insert h*/ p.ins_h, p.ins_w, shift, in_sign, filter_sign,
  //         bias_sign, true, /*mulshift*/ 0, 0, 0, 0,
  //         BM1684::instance()->bdc_node);
  //   }
  // } else {
  //   BM1684::instance().dl_nodechip_conv_forward_local(
  //       in_gi.out_addr, f_gi.out_addr, b_gi.out_addr, gi.out_addr,
  //       gi.buffer_addr, bottom_dim, top_dim, p.groups, p.kh, p.kw, p.dh,
  //       p.dw, pad_h_t, pad_h_b, p.pwl, p.pwr, p.sh, p.sw, p.has_bias ? 1 : 0,
  //       /* result_add*/ 0, p.do_relu ? 1 : 0, p.relu_limit,
  //       unused_ht_for_input, unused_hb_for_input, unused_wl_for_input,
  //       unused_wr_for_input, BM1684::instance()->bdc_node);
  // }
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================

uint32_t tpu::ConvBwdWeightOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  return -1;

  // uint32_t fw_ir_length = 0;
  // ir_layer_info_t *conv_layer_info = (ir_layer_info_t *)ir_layer_info;
  // conv_layer_info->data_size =
  //     get_dynamic_compiler_tensor_datasize(getOutput());
  // conv_layer_info->intensor_store_mode = BM168x::getStoreMode(getInput());
  // conv_layer_info->outtensor_store_mode = BM168x::getStoreMode(getOutput());

  // fw_conv_layer_param_t fw_conv_layer_param = {0};
  // assign_fw_param((void *)&fw_conv_layer_param);

  // conv_layer_info->fw_layer_param_u.fw_conv_layer_param =
  // fw_conv_layer_param; fw_ir_length += sizeof(fw_conv_layer_param_t);

  // return fw_ir_length;
}

int64_t tpu::ConvBwdWeightOp::get_fw_type_bm1684() { return FW_BMNET_CONV; }

// ======================================
// Dynamic LocalGenInterface
// ======================================

int32_t tpu::ConvBwdWeightOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  return -1;
  // int fw_ir_length = 0;
  // ir_layer_info_t *conv_layer_info = (ir_layer_info_t *)ir_layer_info;
  // conv_layer_info->data_size =
  //     get_dynamic_compiler_tensor_datasize(getOutput());
  // conv_layer_info->intensor_store_mode = BM168x::getStoreMode(getInput());
  // conv_layer_info->outtensor_store_mode = BM168x::getStoreMode(getOutput());

  // // get fw layer param
  // fw_conv_layer_param_t fw_conv_layer_param = {0};
  // assign_fw_param((void *)&fw_conv_layer_param);

  // int64_t n, c, h, w;
  // module::getNCHW(getOutput(), n, c, h, w);
  // fw_conv_layer_param.c_idx = 0; // not support c_slice now
  // fw_conv_layer_param.reference_id = get_tensor_id(getOutput());
  // fw_conv_layer_param.concat_c = c;

  // fw_conv_layer_param.if_double_buffer = 0;
  // int using_immbuf = 0; // ref to ConvBwdWeightOp::getBufferSize_bm1684
  // fw_conv_layer_param.if_relu |=
  //     (using_immbuf << 4); // conv's if_relu high 4 bit used as immbuf flag

  // uint64_t weight_global_addr = module::getAddress(getFilter());
  // fw_conv_layer_param.weight_global_offset = weight_global_addr;
  // fw_conv_layer_param.double_buffer_local_offset = 0;

  // conv_layer_info->fw_layer_param_u.fw_conv_layer_param =
  // fw_conv_layer_param; fw_ir_length += sizeof(fw_conv_layer_param_t);

  // // get layer input and output
  // conv_layer_info->ir_tensor_info_v.clear();
  // // input tensor
  // dynamic_push_back_local_tensor(conv_layer_info->ir_tensor_info_v,
  // getInput());

  // // weight
  // dynamic_push_back_local_tensor(conv_layer_info->ir_tensor_info_v,
  //                                getFilter());

  // // bias
  // if (getWithBias()) {
  //   dynamic_push_back_local_tensor(conv_layer_info->ir_tensor_info_v,
  //                                  getBias());
  // }

  // // output, in local concat case, let the out_tensor_id which is the first
  // // conv of concat be concat's out_tensor_id
  // dynamic_push_back_local_tensor(conv_layer_info->ir_tensor_info_v,
  //                                getOutput());

  // // compute fw ir info length for conv input and output
  // fw_ir_length += (sizeof(uint32_t) + 3 * sizeof(uint32_t) +
  //                  (getWithBias() ? 1 : 0) * sizeof(uint32_t));

  // // add fw ir length for output consumer number
  // fw_ir_length += sizeof(uint32_t);

  // return fw_ir_length;
}

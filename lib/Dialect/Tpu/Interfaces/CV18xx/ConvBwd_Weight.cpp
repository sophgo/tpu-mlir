//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/TPUCompressUtil.h"

using namespace tpu_mlir::backend;

// ======================================
// GlobalGenInterface
// ======================================

void tpu::ConvBwdWeightOp::codegen_global_cv18xx(int64_t layer_id) {
  // auto attr = parseParam();
  // gaddr_t ga_input = module::getAddress(getInput());
  // gaddr_t ga_output = module::getAddress(getOutput());
  // gaddr_t ga_filter = module::getAddress(getFilter());
  // gaddr_t ga_pc_info = GA_INVALID;
  // if (module::isUniformQuantized(getOutput()) || attr.has_bias) {
  //   ga_pc_info = module::getAddress(getBias());
  // }

  // auto filterOp = getFilter().getDefiningOp<top::WeightOp>();
  // bool do_compress =
  //     filterOp.getDoCompress().has_value() &&
  //     filterOp.getDoCompress().value();
  // do_compress = attr.groups > 1 ? false : do_compress;
  // WeightCompresser weight_opt(this->getOperation(), do_compress);
  // if (module::isUniformQuantized(getOutput())) {
  //   bool do_ic_alignment = getUse_3icOptimize() ? true : false;
  //   gaddr_t ga_scale_lut = GA_INVALID;
  //   // fuse leakyrelu
  //   int fused_leakyrelu_pos_rshift = 0;
  //   int fused_leakyrelu_pos_m_i8 = 0;
  //   int fused_leakyrelu_neg_rshift = 0;
  //   int fused_leakyrelu_neg_m_i8 = 0;
  //   float fused_negative_slope = 0.0f; // Todo this->do_leaky_relu()

  //   auto do_relu = attr.do_relu;
  //   auto do_leaky_relu = getDoLeakyRelu();
  //   if (do_leaky_relu.has_value() && do_leaky_relu.value()) {
  //     fused_negative_slope =
  //         static_cast<float>(getNegSlope().value().convertToDouble());
  //     fused_leakyrelu_pos_rshift = static_cast<int>(getRshiftPos().value());
  //     fused_leakyrelu_neg_rshift = static_cast<int>(getRshiftNeg().value());
  //     fused_leakyrelu_pos_m_i8 =
  //     static_cast<int>(getMultiplierPos().value()); fused_leakyrelu_neg_m_i8
  //     = static_cast<int>(getMultiplierNeg().value()); do_relu = true;
  //   }

  //   cvi_backend_tg_fixed_conv_kernel(
  //       layer_id,   // layer_id,
  //       ga_input,   // input_data_gaddr,
  //       ga_output,  // output_data_gaddr,
  //       ga_filter,  // weight_data_gaddr,
  //       ga_pc_info, // bias_data_gaddr,
  //       attr.n, attr.ic, attr.ih, attr.iw,
  //       attr.groups, // group,
  //       attr.oc, attr.kh, attr.kw, attr.dh, attr.dw, attr.pht, attr.phb,
  //       attr.pwl,
  //       attr.pwr,               // pad (t, b, l, r)
  //       attr.ins_h, attr.ins_w, // ins_h, ins_w
  //       attr.sh, attr.sw,
  //       attr.has_bias,                             // bias_term,
  //       do_relu ? 1 : 0,                           // do_activation,
  //       do_relu ? &fused_negative_slope : nullptr, // activation_arg,
  //       fused_leakyrelu_pos_m_i8,                  // activation_gt_scale,
  //       fused_leakyrelu_pos_rshift,                // activation_gt_rshift,
  //       fused_leakyrelu_neg_m_i8,                  // activation_le_scale,
  //       fused_leakyrelu_neg_rshift,                // activation_le_rshift,
  //       0,               // (int)rshift[0], //right_shift_width,
  //       true,            // do_chl_quan
  //       do_ic_alignment, // do_ic_alignment,
  //       &weight_opt.old_data, &weight_opt.new_data,
  //       attr.pad_value, // pad_value
  //       ga_scale_lut);
  // } else {
  //   bool do_quant = false;
  //   gaddr_t ga_scale = GA_INVALID;
  //   gaddr_t ga_zeropoint = GA_INVALID;
  //   cvi_backend_tg_bf16_conv_kernel(layer_id,   // layer_id
  //                                   ga_input,   // input_data_gaddr,
  //                                   ga_output,  // output_data_gaddr,
  //                                   ga_filter,  // weight_data_gaddr,
  //                                   ga_pc_info, // bias_data_gaddr,
  //                                   attr.n, attr.ic, attr.ih, attr.iw,
  //                                   attr.groups, // group
  //                                   attr.oc, attr.kh, attr.kw, attr.dh,
  //                                   attr.dw, attr.pht, attr.phb, attr.pwl,
  //                                   attr.pwr,               // pad (t, b, l,
  //                                   r) attr.ins_h, attr.ins_w, // ins_h,
  //                                   ins_w attr.sh, attr.sw, attr.has_bias, //
  //                                   bias_term, attr.do_relu ? 1 : 0, //
  //                                   do_activation, false,                //
  //                                   fp32_output &weight_opt.old_data,
  //                                   &weight_opt.new_data, do_quant, ga_scale,
  //                                   ga_zeropoint); // TODO
  // }
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::ConvBwdWeightOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  // auto do_leaky_relu = getDoLeakyRelu();
  // if (do_leaky_relu.has_value() && do_leaky_relu.value()) {
  //   int64_t n, c, h, w;
  //   auto vOut = getOutput();
  //   module::getNCHW(vOut, n, c, h, w);
  //   auto fmt = CV18xx::getDataType(vOut);
  //   return CV18xx::lmem_woring_size({out_nslice, c, out_hslice, w}, 1, true,
  //                                   fmt);
  // }
  return 0;
}

void tpu::ConvBwdWeightOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                                int64_t d_step, int64_t w_step,
                                                group_type_t group_type,
                                                local_sec_info_t &sec_info,
                                                int64_t layer_id) {
  // auto attr = parseParam();
  // auto gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  // auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  // auto out_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);
  // auto w_gi = LocalGenInterface::getGroupInfo(getFilter());
  // auto b_gi = LocalGenInterface::getGroupInfo(getBias());

  // laddr_t la_input = in_gi.out_addr;
  // laddr_t la_output = out_gi.out_addr;
  // laddr_t la_weight = w_gi.out_addr;
  // laddr_t la_bias = b_gi.out_addr;

  // laddr_t la_working = gi.buffer_addr;
  // bool do_ic_alignment = this->getUse_3icOptimize();

  // int n = sec_info.n_slice;
  // int ih = sec_info.h_slice;
  // int oh = sec_info.out_h_slice;

  // uint32_t pht = (sec_info.h_idx == 0 ? attr.pht : 0);
  // uint32_t phb = (sec_info.h_idx + sec_info.h_slice == attr.ih ? attr.phb :
  // 0);

  // if (module::isUniformQuantized(getOutput())) {
  //   float neg_slope = 0.0f;
  //   int8_t pos_rshift = 0, pos_m_i8 = 0;
  //   int8_t neg_rshift = 0, neg_m_i8 = 0;

  //   auto do_leaky_relu = getDoLeakyRelu();
  //   if (do_leaky_relu.has_value() && do_leaky_relu.value()) {
  //     neg_slope =
  //     static_cast<float>(getNegSlope().value().convertToDouble()); pos_rshift
  //     = static_cast<int8_t>(getRshiftPos().value()); neg_rshift =
  //     static_cast<int8_t>(getRshiftNeg().value()); pos_m_i8 =
  //     static_cast<int8_t>(getMultiplierPos().value()); neg_m_i8 =
  //     static_cast<int8_t>(getMultiplierNeg().value());
  //   }
  //   cvi_backend_tl_conv(layer_id, la_input, la_output, la_weight, la_working,
  //                       la_bias, n, attr.ic, ih, attr.iw, attr.groups,
  //                       attr.oc, oh, attr.ow, attr.kh, attr.kw, attr.dh,
  //                       attr.dw, pht, phb, attr.pwl, attr.pwr, attr.sh,
  //                       attr.sw, attr.ins_h, attr.ins_w, 0, /*result_add*/ 0,
  //                       /*crtl*/ attr.has_bias, attr.do_relu, neg_slope, 0,
  //                       /*rshift*/ attr.oc,    /*rshift_shift_len*/
  //                       pos_rshift, /*rshift_pos*/
  //                       neg_rshift, /*rshift8_neg*/
  //                       pos_m_i8,   /*m_i8_pos*/
  //                       neg_m_i8,   /*m_i8_neg*/
  //                       do_ic_alignment);
  // } else {
  //   cvi_backend_bf16_tl_conv(
  //       layer_id, la_input, la_output, la_weight, la_working, la_bias, n,
  //       attr.ic, ih, attr.iw, attr.groups, attr.oc, oh, attr.ow, attr.kh,
  //       attr.kw, attr.dh, attr.dw, pht, phb, attr.pwl, attr.pwr, attr.sh,
  //       attr.sw, attr.ins_h, attr.ins_w, attr.has_bias, attr.do_relu);
  // }
  return;
}

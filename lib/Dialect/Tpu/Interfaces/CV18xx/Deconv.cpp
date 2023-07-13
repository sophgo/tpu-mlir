//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Deconv.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

// ======================================
// GlobalGenInterface
// ======================================

void tpu::DeconvOp::codegen_global_cv18xx(int64_t layer_id) {

  auto attr = parseParam();
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  gaddr_t ga_filter = module::getAddress(getFilter());
  gaddr_t ga_pc_info = GA_INVALID;
  if (module::isUniformQuantized(getOutput()) || attr.with_bias) {
    ga_pc_info = module::getAddress(getBias());
  }

  int kh_ext = (attr.kh - 1) * attr.dh + 1;
  int kw_ext = (attr.kw - 1) * attr.dw + 1;
  int ins_h = attr.sh - 1;
  int ins_w = attr.sw - 1;
  int pad_t = kh_ext - attr.pad_h - 1;
  int pad_l = kw_ext - attr.pad_w - 1;
  int pad_b = attr.oh + attr.pad_h - (attr.ih - 1) * attr.sh - 1;
  int pad_r = attr.ow + attr.pad_w - (attr.iw - 1) * attr.sw - 1;
  int sh = 1;
  int sw = 1;

  if (module::isUniformQuantized(getOutput())) {
    cvi_backend_tg_fixed_conv_kernel(layer_id,   // layer_id,
                                     ga_input,   // input_data_gaddr,
                                     ga_output,  // output_data_gaddr,
                                     ga_filter,  // weight_data_gaddr,
                                     ga_pc_info, // bias_data_gaddr,
                                     attr.n, attr.ic, attr.ih, attr.iw,
                                     attr.g, // group,
                                     attr.oc, attr.kh, attr.kw, attr.dh,
                                     attr.dw, pad_t, pad_b, pad_l, pad_r, ins_h,
                                     ins_w, sh, sw,
                                     attr.with_bias,       // bias_term,
                                     attr.do_relu ? 1 : 0, // do_activation,
                                     nullptr,              // activation_arg,
                                     0, // activation_gt_scale,
                                     0, // activation_gt_rshift,
                                     0, // activation_le_scale,
                                     0, // activation_le_rshift,
                                     0, // (int)rshift[0], //right_shift_width,
                                     true, // do_chl_quan
                                     false // do_ic_alignment,
    );
  } else {
    cvi_backend_tg_bf16_conv_kernel(layer_id,   // layer_id
                                    ga_input,   // input_data_gaddr,
                                    ga_output,  // output_data_gaddr,
                                    ga_filter,  // weight_data_gaddr,
                                    ga_pc_info, // bias_data_gaddr,
                                    attr.n, attr.ic, attr.ih, attr.iw,
                                    attr.g, // group
                                    attr.oc, attr.kh, attr.kw, attr.dh, attr.dw,
                                    pad_t, pad_b, pad_l, pad_r, ins_h, ins_w,
                                    sh, sw,
                                    attr.with_bias,       // bias_term,
                                    attr.do_relu ? 1 : 0, // do_activation,
                                    false                 // fp32_output
    );
  }
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::DeconvOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::DeconvOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                         int64_t d_step, int64_t w_step,
                                         group_type_t group_type,
                                         local_sec_info_t &sec_info,
                                         int64_t layer_id) {
  auto attr = parseParam();
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto out_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);
  auto w_gi = LocalGenInterface::getGroupInfo(getFilter());
  auto b_gi = LocalGenInterface::getGroupInfo(getBias());

  laddr_t la_input = in_gi.out_addr;
  laddr_t la_output = out_gi.out_addr;
  laddr_t la_weight = w_gi.out_addr;
  laddr_t la_bias = b_gi.out_addr;

  bool do_ic_alignment = false;

  int n = sec_info.n_slice;
  int ih = sec_info.h_slice;
  int oh = sec_info.out_h_slice;

  int pad_h_top, pad_h_bottom;
  int pad_w_left, pad_w_right;
  int kh_ext = (attr.kh - 1) * attr.dh + 1;
  int kw_ext = (attr.kw - 1) * attr.dw + 1;
  int ins_last_h = 0;
  int ins_last_w = (attr.ow + attr.pad_w + attr.pad_w_after - kw_ext) % attr.sw;
  int ins_h = attr.sh - 1;
  int ins_w = attr.sw - 1;
  if (auto deconv_in_slice =
          DeconvSlice(sec_info.out_h_idx, sec_info.out_h_slice, attr.sh, kh_ext,
                      attr.ih, attr.pad_h)) {
    pad_h_top = deconv_in_slice.value()[0];
    pad_h_bottom = deconv_in_slice.value()[1];

  } else {
    pad_h_top = attr.kh - attr.pad_h - 1;
    pad_h_bottom = attr.kh - attr.pad_h_after - 1;
  }
  pad_w_left = attr.kw - attr.pad_w - 1;
  pad_w_right = attr.kw - attr.pad_w_after - 1;
  // hw limitation once set ins_w / ins_h that input w/h should > 1
  if (ins_h && ih == 1) {
    ins_last_h += ins_h;
    ins_h = 0;
    if (pad_h_top) {
      ins_last_h = 0; // included in pad_h_top
    }
  }
  if (ins_w && attr.iw == 1) {
    ins_last_w += ins_w;
    ins_w = 0;
    if (pad_w_left) {
      // TODO: need to verify
      ins_last_w = 0; // included in pad_w_left
    }
  }
  assert(ins_last_h < 16 && ins_last_w < 16);
  assert(pad_h_top < 16 && pad_h_bottom < 16 && pad_w_left < 16 &&
         pad_w_right < 16);
  if (module::isUniformQuantized(getOutput())) {
    cvi_backend_tl_deconv(
        layer_id, la_input, la_output, la_weight, la_bias, n, attr.ic, ih,
        attr.iw, attr.g, attr.oc, oh, attr.ow, attr.kh, attr.kw, attr.dh,
        attr.dw, ins_h, ins_last_h, ins_w, ins_last_w, pad_h_top, pad_h_bottom,
        pad_w_left, pad_w_right, attr.sh, attr.sw, attr.with_bias,
        getDoRelu(), // do_activation,
        0,           // right_shift_width,
        attr.oc,     // right_shift_array_len
        do_ic_alignment);
  } else {
    cvi_backend_tl_bf16_deconv(
        layer_id, la_input, la_output, la_weight, la_bias, n, attr.ic, ih,
        attr.iw, attr.g, attr.oc, oh, attr.ow, attr.kh, attr.kw, attr.dh,
        attr.dw, ins_h, ins_last_h, ins_w, ins_last_w, pad_h_top, pad_h_bottom,
        pad_w_left, pad_w_right, attr.sh, attr.sw, attr.with_bias, getDoRelu());
  }
  return;
}

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "tl_conv"

namespace tpu_mlir {
namespace backend {
typedef struct _conv_args {
  uint32_t n;
  uint32_t ic;
  uint32_t ih;
  uint32_t iw;
  uint32_t g;
  uint32_t oc;
  uint32_t oh;
  uint32_t ow;
  uint16_t kh;
  uint16_t kw;
  uint8_t dh;
  uint8_t dw;
  uint8_t pad_t;
  uint8_t pad_b;
  uint8_t pad_l;
  uint8_t pad_r;
  uint8_t sh;
  uint8_t sw;
  uint8_t ins_h;
  uint8_t ins_w;
  bool result_add;
  bool with_bias;
  bool do_relu;
} conv_args_t;

//
// This function implemnets conv load all slice oc.
//   - input and output load once and no slicing
//   - weight load on sliced oc_step, no slicing on ic
//   - 2x weight buffer for double buffer parallel
//
static void conv_la_oc_step(uint32_t layer_id, gaddr_t ga_ifmap,
                            gaddr_t ga_ofmap, gaddr_t ga_filter,
                            gaddr_t ga_perchannel, const conv_args_t &args,
                            uint32_t oc_step) {
  assert(oc_step % CV18xx::NPU_NUM == 0);

  uint32_t n = args.n;
  uint32_t ic = args.ic;
  uint32_t ih = args.ih;
  uint32_t iw = args.iw;
  uint32_t g = args.g;
  uint32_t oc = args.oc;
  uint16_t kh = args.kh;
  uint16_t kw = args.kw;
  int dh = args.dh;
  int dw = args.dw;
  int pad_t = args.pad_t;
  int pad_b = args.pad_b;
  int pad_l = args.pad_l;
  int pad_r = args.pad_r;
  int sh = args.sh;
  int sw = args.sw;
  int ins_h = args.ins_h;
  int ins_w = args.ins_w;
  uint16_t kh_ext = dh * (kh - 1) + 1;
  uint16_t kw_ext = dw * (kw - 1) + 1;
  uint32_t oh = (ih + pad_t + pad_b - kh_ext) / sh + 1;
  uint32_t ow = (iw + pad_l + pad_r - kw_ext) / sw + 1;
  assert(oh == args.oh);
  assert(ow == args.ow);
  bool result_add = args.result_add;
  bool with_bias = args.with_bias;
  bool do_relu = args.do_relu;
  assert(result_add == false);
  bool is_dw = false;
  if (g != 1 && g == oc && g == ic) {
    is_dw = true;
  } else {
    assert(g == 1);
  }

  LLVM_DEBUG(llvm::errs() << "    conv_la_oc_step: nchw = (" << n << "," << ic
                          << "," << ih << "," << iw
                          << ")"
                             ", ochw = ("
                          << oc << "," << oh << "," << ow
                          << ")"
                             ", g = "
                          << g << "\n                     "
                          << "K: " << kh << "*" << kw << ", D: " << dh << "*"
                          << dw << ", P: " << pad_t << "," << pad_l << " <-> "
                          << pad_b << "," << pad_r << ", S: " << sh << "*" << sw
                          << "\n                     "
                          << "result_add: " << result_add << ", with_bias: "
                          << with_bias << ", do_relu: " << do_relu
                          << ", is_dw: " << is_dw << "\n";);

  // alloc lmem_t
  cvk_tl_t *tl_ifmap = nullptr;
  cvk_tl_t *tl_ofmap = nullptr;
  cvk_tl_t *tl_filter[2] = {nullptr, nullptr};
  cvk_tl_t *tl_perchannel = nullptr;

  // global memory stride from global memory shape
  cvk_tg_stride_t ifmap_gstride = {ic * ih * iw, ih * iw, iw};
  cvk_tg_stride_t ofmap_gstride = {oc * oh * ow, oh * ow, ow};
  // filter has been transposed from (oc, ic, kh, kw) to (1, oc, kh * kw, ic)
  cvk_tg_stride_t filter_gstride = {oc * kh * kw * ic / g, kh * kw * ic / g,
                                    ic / g};

  // Pre-alloc maximum one-step size
  tl_ifmap = CV18xx::lmem_alloc_tensor(CV18xx::tl_shape_t4(n, ic, ih, iw),
                                       CVK_FMT_I8, /*eu_align=*/1);
  tl_ofmap = CV18xx::lmem_alloc_tensor(CV18xx::tl_shape_t4(n, oc, oh, ow),
                                       CVK_FMT_I8, /*eu_align=*/1);
  tl_filter[0] = CV18xx::lmem_alloc_tensor(
      CV18xx::tl_shape_t4(ic / g, oc_step, kh, kw), CVK_FMT_I8, /*eu_align=*/0);
  tl_filter[1] = CV18xx::lmem_alloc_tensor(
      CV18xx::tl_shape_t4(ic / g, oc_step, kh, kw), CVK_FMT_I8, /*eu_align=*/0);
  int perchannel_size = CV18xx::chan_quan_param_size(with_bias);
  tl_perchannel =
      CV18xx::lmem_alloc_tensor(CV18xx::tl_shape_t4(1, oc, 1, perchannel_size),
                                CVK_FMT_U8, /*eu_align=*/0);
  assert(tl_filter[0] && tl_filter[1] && tl_ifmap && tl_ofmap && tl_perchannel);

  // bmk does not keep eu-align info, user need to update stride if shape
  // changed
  tl_ifmap->shape = CV18xx::tl_shape_t4(n, ic, ih, iw);
  tl_ifmap->stride =
      CV18xx::tl_default_stride(tl_ifmap->shape, CVK_FMT_I8, /*eu_align=*/1);
  CV18xx::tdma_load_stride(tl_ifmap, ga_ifmap, ifmap_gstride);

  tl_perchannel->shape = CV18xx::tl_shape_t4(1, oc, 1, perchannel_size);
  tl_perchannel->stride = CV18xx::tl_default_stride(tl_perchannel->shape,
                                                    CVK_FMT_I8, /*eu_aign=*/0);
  CV18xx::parallel_disable();
  CV18xx::tdma_load(tl_perchannel, ga_perchannel);
  CV18xx::parallel_enable();

  tl_ofmap->stride = CV18xx::tl_default_stride(tl_ofmap->shape, CVK_FMT_I8, 1);
  // split oc
  int flip = 0;
  for (uint32_t n_pos = 0; n_pos < n; n_pos++) {
    cvk_tl_t tl_ifmap_step;
    tl_ifmap_step.start_address =
        tl_ifmap->start_address + n_pos * tl_ifmap->stride.n;
    tl_ifmap_step.fmt = CVK_FMT_I8;
    tl_ifmap_step.shape = CV18xx::tl_shape_t4(1, ic, ih, iw);
    tl_ifmap_step.stride = CV18xx::tl_default_stride(
        tl_ifmap_step.shape, CVK_FMT_I8, /*eu_align=*/1);

    for (uint32_t oc_pos = 0; oc_pos < oc; oc_pos += oc_step) {
      uint32_t cur_oc = std::min(oc - oc_pos, oc_step);

      uint64_t ga_filter_oc_pos = ga_filter + oc_pos * kh * kw * ic / g;
      cvk_tl_t tl_tmp;
      tl_tmp.start_address = tl_filter[flip]->start_address;
      tl_tmp.fmt = CVK_FMT_I8;
      tl_tmp.shape = CV18xx::tl_shape_t4(1, cur_oc, kh * kw, ic / g);
      tl_tmp.stride =
          CV18xx::tl_default_stride(tl_tmp.shape, CVK_FMT_I8, /*eu_align=*/0);
      CV18xx::parallel_disable();
      CV18xx::tdma_load_stride(&tl_tmp, ga_filter_oc_pos, filter_gstride);
      CV18xx::parallel_enable();

      // TODO: this looks weird
      // Reshape per channel quantization data for TIU
      // tl_perchannel->shape = CV18xx::tl_shape_t4(1, cur_oc, 1, 1);
      // tl_perchannel->stride = CV18xx::tl_default_stride(tl_perchannel->shape,
      // CVK_FMT_I8, /*eu_aign=*/0);
      cvk_tl_t tl_perchannel_oc_pos;
      tl_perchannel_oc_pos.start_address =
          tl_perchannel->start_address +
          (oc_pos / CV18xx::NPU_NUM) * perchannel_size;
      tl_perchannel_oc_pos.fmt = CVK_FMT_I8;
      tl_perchannel_oc_pos.shape =
          CV18xx::tl_shape_t4(1, cur_oc, 1, 1); // 1? not 9 or 5? very strange
      tl_perchannel_oc_pos.stride = CV18xx::tl_default_stride(
          tl_perchannel_oc_pos.shape, CVK_FMT_I8, /*eu_align=*/0);

      // Adjust current shape and stride
      // bmk does not keep eu-align info, user need to update stride if shape
      // changed
      // tl_ofmap->shape = CV18xx::tl_shape_t4(n, cur_oc, oh, ow);
      // tl_ofmap->stride = CV18xx::tl_default_stride(tl_ofmap->shape,
      // CVK_FMT_I8, /*eu_aign=*/1);
      cvk_tl_t tl_ofmap_oc_pos;
      uint32_t ofmap_count = align_up(oh * ow, CV18xx::EU_BYTES);
      tl_ofmap_oc_pos.start_address = tl_ofmap->start_address +
                                      n_pos * tl_ofmap->stride.n +
                                      (oc_pos / CV18xx::NPU_NUM) * ofmap_count;
      tl_ofmap_oc_pos.fmt = CVK_FMT_I8;
      tl_ofmap_oc_pos.shape = CV18xx::tl_shape_t4(1, cur_oc, oh, ow);
      tl_ofmap_oc_pos.stride = CV18xx::tl_default_stride(
          tl_ofmap_oc_pos.shape, CVK_FMT_I8, /*eu_align=*/1);

      if (!is_dw) {
        // filter shape for tiu != shape for load
        // bmk does not keep eu-align info, user need to update stride if shape
        // changed
        tl_filter[flip]->shape = CV18xx::tl_shape_t4(ic / g, cur_oc, kh, kw);
        tl_filter[flip]->stride = CV18xx::tl_default_stride(
            tl_filter[flip]->shape, CVK_FMT_I8, /*eu_aign*/ 0);

        cvk_tiu_convolution_param_t param = {0};
        param.ofmap = &tl_ofmap_oc_pos;
        param.ifmap = &tl_ifmap_step;
        param.weight = tl_filter[flip];
        param.chl_quan_param = &tl_perchannel_oc_pos;
        param.ins_h = ins_h;
        param.ins_w = ins_w;
        param.ins_last_h = 0;
        param.ins_last_w = 0;
        param.pad_top = pad_t;
        param.pad_bottom = pad_b;
        param.pad_left = pad_l;
        param.pad_right = pad_r;
        param.stride_h = sh;
        param.stride_w = sw;
        param.dilation_h = dh;
        param.dilation_w = dw;
        param.has_bias = with_bias ? 1 : 0;
        param.relu_enable = do_relu ? 1 : 0;
        param.ps32_mode = 0;
        param.w_is_const = 0;
        param.layer_id = layer_id;
        param.ins_val = 0; // symmetric quantization
        param.ins_fp =
            CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization

        CV18xx::tiu_convolution(&param);
      } else {
        cvk_tl_t tl_ifmap_oc_pos;
        uint32_t ifmap_count = align_up(ih * iw, CV18xx::EU_BYTES);
        tl_ifmap_oc_pos.start_address =
            tl_ifmap_step.start_address +
            (oc_pos / CV18xx::NPU_NUM) * ifmap_count;
        tl_ifmap_oc_pos.fmt = CVK_FMT_I8;
        tl_ifmap_oc_pos.shape = CV18xx::tl_shape_t4(1, cur_oc, ih, iw);
        tl_ifmap_oc_pos.stride = CV18xx::tl_default_stride(
            tl_ifmap_oc_pos.shape, CVK_FMT_I8, /*eu_align=*/1);

        // filter shape for tiu != shape for load
        // bmk does not keep eu-align info, user need to update stride if shape
        // changed
        tl_filter[flip]->shape = CV18xx::tl_shape_t4(1, cur_oc, kh, kw);
        tl_filter[flip]->stride = CV18xx::tl_default_stride(
            tl_filter[flip]->shape, CVK_FMT_I8, /*eu_aign*/ 1);

        cvk_tiu_depthwise_convolution_param_t param = {0};
        param.ofmap = &tl_ofmap_oc_pos;
        param.ifmap = &tl_ifmap_oc_pos;
        param.weight = tl_filter[flip];
        param.chl_quan_param = &tl_perchannel_oc_pos;
        param.ins_h = ins_h;
        param.ins_w = ins_w;
        param.ins_last_h = 0;
        param.ins_last_w = 0;
        param.pad_top = pad_t;
        param.pad_bottom = pad_b;
        param.pad_left = pad_l;
        param.pad_right = pad_r;
        param.stride_h = sh;
        param.stride_w = sw;
        param.dilation_h = dh;
        param.dilation_w = dw;
        param.has_bias = with_bias ? 1 : 0;
        param.relu_enable = do_relu ? 1 : 0;
        param.layer_id = layer_id;
        param.ins_val = 0; // symmetric quantization
        param.ins_fp =
            CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization

        CV18xx::tiu_depthwise_convolution(&param);
      }
      flip = 1 - flip;
    }
  }

  tl_ofmap->shape = CV18xx::tl_shape_t4(n, oc, oh, ow);
  tl_ofmap->stride =
      CV18xx::tl_default_stride(tl_ofmap->shape, CVK_FMT_I8, /*eu_align=*/1);
  CV18xx::parallel_disable();
  CV18xx::tdma_store_stride(tl_ofmap, ga_ofmap, ofmap_gstride);
  CV18xx::parallel_enable();

  //
  // Release resource in reverse order
  //
  CV18xx::lmem_free_tensor(tl_perchannel);
  CV18xx::lmem_free_tensor(tl_filter[1]);
  CV18xx::lmem_free_tensor(tl_filter[0]);
  CV18xx::lmem_free_tensor(tl_ofmap);
  CV18xx::lmem_free_tensor(tl_ifmap);
}

// assuming in-place, i.e. input and output at same address
static void tl_leaky_relu(uint32_t layer_id,
                          cvk_tl_t &relu, // both input and output
                          cvk_tl_t &working, int8_t rshift_pos, int8_t m_i8_pos,
                          int8_t rshift_neg, int8_t m_i8_neg) {
  bool isIgnorePosPart = (m_i8_pos == 0 || (m_i8_pos == 1 && rshift_pos == 0));
  bool isSlopeSmallerThanOne = ((m_i8_neg >> rshift_neg) == 0);

  if (isIgnorePosPart && m_i8_neg >= 0) {
    cvk_tiu_mul_param_t p4 = {0};
    p4.res_high = nullptr;
    p4.res_low = &working;
    p4.a = &relu;
    p4.b_const.val = m_i8_neg;
    p4.b_const.is_signed = true;
    p4.b_is_const = 1;
    p4.rshift_bits = rshift_neg;
    p4.layer_id = layer_id;
    p4.relu_enable = 0;
    CV18xx::tiu_mul(&p4);

    if (isSlopeSmallerThanOne) {
      cvk_tiu_max_param_t p1 = {0};
      p1.max = &relu;
      p1.a = &relu;
      p1.b = &working;
      p1.b_is_const = 0;
      p1.layer_id = layer_id;
      CV18xx::tiu_max(&p1);
    } else {
      cvk_tiu_min_param_t p1 = {0};
      p1.min = &relu;
      p1.a = &relu;
      p1.b = &working;
      p1.b_is_const = 0;
      p1.layer_id = layer_id;
      CV18xx::tiu_min(&p1);
    }
  } else {
    // 0. pos -> working
    cvk_tiu_max_param_t p13 = {0};
    p13.max = &working;
    p13.a = &relu;
    p13.b_is_const = 1;
    p13.b_const.is_signed = 1;
    p13.b_const.val = 0;
    p13.layer_id = layer_id;
    CV18xx::tiu_max(&p13);

    if (!isIgnorePosPart) {
      // 1. working(pos) apply pos rshift and m_i8
      cvk_tiu_mul_param_t p = {0};
      p.res_high = nullptr;
      p.res_low = &working;
      p.a = &working;
      p.b_const.val = m_i8_pos;
      p.b_const.is_signed = true;
      p.b_is_const = 1;
      p.rshift_bits = rshift_pos;
      p.layer_id = layer_id;
      p.relu_enable = 0;
      CV18xx::tiu_mul(&p);
    }

    // 2. neg -> relu (in-place)
    cvk_tiu_min_param_t p7 = {0};
    p7.min = &relu;
    p7.a = &relu;
    p7.b_is_const = 1;
    p7.b_const.val = 0;
    p7.b_const.is_signed = 1;
    p7.layer_id = layer_id;
    CV18xx::tiu_min(&p7);

    // 3. relu (in-place) apply neg rshift and m_i8
    cvk_tiu_mul_param_t p8 = {0};
    p8.res_high = nullptr;
    p8.res_low = &relu;
    p8.a = &relu;
    p8.b_const.val = m_i8_neg;
    p8.b_const.is_signed = true;
    p8.b_is_const = 1;
    p8.rshift_bits = rshift_neg;
    p8.layer_id = layer_id;
    p8.relu_enable = 0;
    CV18xx::tiu_mul(&p8);

    // 4. relu(neg) = or working(pos)
    cvk_tiu_or_int8_param_t p9 = {0};
    p9.res = &relu;
    p9.a = &relu;
    p9.b = &working;
    p9.layer_id = layer_id;
    CV18xx::tiu_or_int8(&p9);
  }
}

//
// This function implemnets conv load weight slice oc.
//   - input and output no slicing and been loaded already
//   - weight load on sliced oc_step, no slicing on ic
//   - 2x weight buffer for double buffer parallel
//
static void conv_lw_oc_step(uint32_t layer_id, laddr_t la_ifmap,
                            laddr_t la_ofmap, laddr_t la_working,
                            gaddr_t ga_filter, gaddr_t ga_perchannel,
                            const conv_args_t &args, uint32_t oc_step,
                            bool do_store, gaddr_t ga_ofmap, bool do_leaky_relu,
                            int8_t rshift_pos, int8_t m_i8_pos,
                            int8_t rshift_neg, int8_t m_i8_neg,
                            bool compressed_weight) {
  assert(oc_step % CV18xx::NPU_NUM == 0);

  uint32_t n = args.n;
  uint32_t ic = args.ic;
  uint32_t ih = args.ih;
  uint32_t iw = args.iw;
  uint32_t g = args.g;
  uint32_t oc = args.oc;
  uint16_t kh = args.kh;
  uint16_t kw = args.kw;
  int dh = args.dh;
  int dw = args.dw;
  int pad_t = args.pad_t;
  int pad_b = args.pad_b;
  int pad_l = args.pad_l;
  int pad_r = args.pad_r;
  int sh = args.sh;
  int sw = args.sw;
  int ins_h = args.ins_h;
  int ins_w = args.ins_w;
  uint16_t kh_ext = dh * (kh - 1) + 1;
  uint16_t kw_ext = dw * (kw - 1) + 1;
  uint32_t oh =
      (((ih - 1) * (ins_h + 1) + 1) + pad_t + pad_b - kh_ext) / sh + 1;
  uint32_t ow =
      (((iw - 1) * (ins_w + 1) + 1) + pad_l + pad_r - kw_ext) / sw + 1;
  assert(oh == args.oh);
  assert(ow == args.ow);
  bool result_add = args.result_add;
  bool with_bias = args.with_bias;
  bool do_relu = args.do_relu;
  assert(result_add == false);
  bool is_dw = false;
  if (g != 1 && g == oc && g == ic) {
    is_dw = true;
  } else {
    assert(g == 1);
  }

  LLVM_DEBUG(
      llvm::errs() << "    conv_lw_oc_step: nchw = (" << n << "," << ic << ","
                   << ih << "," << iw
                   << ")"
                      ", ochw = ("
                   << oc << "," << oh << "," << ow
                   << ")"
                      ", g = "
                   << g << "\n                     "
                   << "K: " << kh << "*" << kw << ", D: " << dh << "*" << dw
                   << ", P: " << pad_t << "," << pad_l << " <-> " << pad_b
                   << "," << pad_r << ", S: " << sh << "*" << sw
                   << "\n                     "
                   << "result_add: " << result_add
                   << ", with_bias: " << with_bias << ", do_relu: " << do_relu
                   << ", is_dw: " << is_dw << "\n                     "
                   << "la_i = " << la_ifmap << ", la_o = " << la_ofmap
                   << ", la_w = " << la_working << "\n                     "
                   << "leaky = " << do_leaky_relu << ", rshift/m_i8 = pos("
                   << (int)rshift_pos << "," << (int)m_i8_pos << "),neg("
                   << (int)rshift_neg << "," << (int)m_i8_neg << ")"
                   << "\n                     "
                   << "ST " << do_store
                   << ", ga_o = " << (do_store ? ga_ofmap : 0) << "\n";);

  // allocate working lmem
  // laddr_t la_perchannel = la_working;
  int perchannel_size = CV18xx::chan_quan_param_size(with_bias);
  // uint32_t ls_perchannel = ceiling_func(oc, CV18xx::NPU_NUM) *
  // perchannel_size;
  laddr_t la_perChannel[2];
  uint32_t ls_perChannel =
      align_up(ceiling_func(oc_step, CV18xx::NPU_NUM) * perchannel_size,
               CV18xx::EU_BYTES);
  la_perChannel[0] = la_working;
  la_perChannel[1] =
      align_up(la_perChannel[0] + ls_perChannel, CV18xx::EU_BYTES);
  // double ping-pong buffer
  laddr_t la_filter[2];
  uint32_t ls_filter =
      ceiling_func(oc_step, CV18xx::NPU_NUM) * kh * kw * ic / g;
  la_filter[0] = align_up(la_working + 2 * ls_perChannel, CV18xx::EU_BYTES);
  la_filter[1] = align_up(la_filter[0] + ls_filter, CV18xx::EU_BYTES);

  // prepare tl_ifmap from la_ifmap
  cvk_tl_t tl_ifmap;
  tl_ifmap.start_address = la_ifmap;
  tl_ifmap.fmt = CVK_FMT_I8;
  tl_ifmap.shape = CV18xx::tl_shape_t4(n, ic, ih, iw);
  tl_ifmap.stride =
      CV18xx::tl_default_stride(tl_ifmap.shape, CVK_FMT_I8, /*eu_align=*/1);

  cvk_tl_t tl_ofmap;
  tl_ofmap.start_address = la_ofmap;
  tl_ofmap.fmt = CVK_FMT_I8;
  tl_ofmap.shape = CV18xx::tl_shape_t4(n, oc, oh, ow);
  tl_ofmap.stride =
      CV18xx::tl_default_stride(tl_ofmap.shape, CVK_FMT_I8, /*eu_align=*/1);

  // split oc
  int flip = 0;
  bool first = true;
  uint32_t oc_pos_flip_back = 0; // for do_stroe
  uint32_t cur_oc_flip_back = 0; // for do_store

  cvk_tl_t tl_ifmap_step;
  tl_ifmap_step.start_address = tl_ifmap.start_address;
  tl_ifmap_step.fmt = CVK_FMT_I8;
  tl_ifmap_step.shape = CV18xx::tl_shape_t4(n, ic, ih, iw);
  tl_ifmap_step.stride = CV18xx::tl_default_stride(tl_ifmap_step.shape,
                                                   CVK_FMT_I8, /*eu_align=*/1);
  for (uint32_t oc_pos = 0; oc_pos < oc; oc_pos += oc_step) {
    uint32_t cur_oc = std::min(oc - oc_pos, oc_step);
    cvk_tl_t tl_filter;
    tl_filter.start_address = la_filter[flip];
    tl_filter.fmt = CVK_FMT_I8;
    tl_filter.shape = CV18xx::tl_shape_t4(1, cur_oc, kh * kw, ic / g);
    tl_filter.stride =
        CV18xx::tl_default_stride(tl_filter.shape, CVK_FMT_I8, /*eu_align=*/0);
    uint64_t ga_filter_oc_pos = ga_filter + oc_pos * kh * kw * ic / g;
    // filter has been transposed from (oc, ic / g, kh, kw) to (1, oc, kh * kw,
    // ic / g)
    cvk_tg_stride_t filter_gstride = {oc * kh * kw * ic / g, kh * kw * ic / g,
                                      ic / g};
    CV18xx::parallel_disable();
    if (!compressed_weight) {
      // Normal weight
      CV18xx::tdma_load_stride(&tl_filter, ga_filter_oc_pos, filter_gstride);
    } else {
      // Compressed weight
      cvk_cmpr_tg_t ts_data = {0};
      ts_data.t.base_reg_index =
          CV18xx::getTdmaBaseSelectIndexFromGaddr(ga_filter_oc_pos);
      ts_data.t.start_address = ga_filter_oc_pos;
      ts_data.t.fmt = tl_filter.fmt;
      ts_data.t.shape = {tl_filter.shape.n, tl_filter.shape.c,
                         tl_filter.shape.h, tl_filter.shape.w};
      ts_data.t.stride = filter_gstride;
      cvk_tdma_g2l_tensor_copy_decompressed_param_t param = {0};
      param.src = &ts_data;
      param.dst = &tl_filter;
      CV18xx::tdma_g2l_tensor_copy_decompressed(&param);
    }

    // load perchannel all at once
    uint64_t ga_perChannel_oc_pos = ga_perchannel + oc_pos * perchannel_size;
    cvk_tl_t tl_perchannel;
    tl_perchannel.start_address = la_perChannel[flip];
    tl_perchannel.fmt = CVK_FMT_I8;
    tl_perchannel.shape = CV18xx::tl_shape_t4(1, cur_oc, 1, perchannel_size);
    tl_perchannel.stride = CV18xx::tl_default_stride(tl_perchannel.shape,
                                                     CVK_FMT_I8, /*eu_aign=*/0);
    CV18xx::tdma_load(&tl_perchannel, ga_perChannel_oc_pos);
    CV18xx::parallel_enable();

    // TODO: this looks weird
    // Reshape per channel quantization data for TIU
    // tl_perchannel->shape = CV18xx::tl_shape_t4(1, cur_oc, 1, 1);
    // tl_perchannel->stride = CV18xx::tl_default_stride(tl_perchannel->shape,
    // CVK_FMT_I8, /*eu_aign=*/0);
    cvk_tl_t tl_perchannel_oc_pos;
    tl_perchannel_oc_pos.start_address = la_perChannel[flip];
    tl_perchannel_oc_pos.fmt = CVK_FMT_I8;
    tl_perchannel_oc_pos.shape =
        CV18xx::tl_shape_t4(1, cur_oc, 1, 1); // 1? not 9 or 5? very strange
    tl_perchannel_oc_pos.stride = CV18xx::tl_default_stride(
        tl_perchannel_oc_pos.shape, CVK_FMT_I8, /*eu_align=*/0);
    // Adjust current shape and stride
    // bmk does not keep eu-align info, user need to update stride if shape
    // changed
    // tl_ofmap->shape = CV18xx::tl_shape_t4(n, cur_oc, oh, ow);
    // tl_ofmap->stride = CV18xx::tl_default_stride(tl_ofmap->shape, CVK_FMT_I8,
    // /*eu_aign=*/1);
    cvk_tl_t tl_ofmap_oc_pos;
    uint32_t ofmap_count = align_up(oh * ow, CV18xx::EU_BYTES);
    tl_ofmap_oc_pos.start_address =
        tl_ofmap.start_address + (oc_pos / CV18xx::NPU_NUM) * ofmap_count;
    tl_ofmap_oc_pos.fmt = CVK_FMT_I8;
    tl_ofmap_oc_pos.shape = CV18xx::tl_shape_t4(n, cur_oc, oh, ow);
    tl_ofmap_oc_pos.stride = CV18xx::tl_default_stride(
        tl_ofmap_oc_pos.shape, CVK_FMT_I8, /*eu_align=*/1);
    tl_ofmap_oc_pos.stride.n =
        tl_ofmap_oc_pos.stride.c * ceiling_func(oc, CV18xx::NPU_NUM);
    if (!is_dw) {
      // tl_filter shape for tiu != shape for load
      tl_filter.shape = CV18xx::tl_shape_t4(ic / g, cur_oc, kh, kw);
      tl_filter.stride =
          CV18xx::tl_default_stride(tl_filter.shape, CVK_FMT_I8, /*eu_aign*/ 0);
      cvk_tiu_convolution_param_t param = {0};
      param.ofmap = &tl_ofmap_oc_pos;
      param.ifmap = &tl_ifmap_step;
      param.weight = &tl_filter;
      param.chl_quan_param = &tl_perchannel_oc_pos;
      param.ins_h = ins_h;
      param.ins_w = ins_w;
      param.ins_last_h = 0;
      param.ins_last_w = 0;
      param.pad_top = pad_t;
      param.pad_bottom = pad_b;
      param.pad_left = pad_l;
      param.pad_right = pad_r;
      param.stride_h = sh;
      param.stride_w = sw;
      param.dilation_h = dh;
      param.dilation_w = dw;
      param.has_bias = with_bias ? 1 : 0;
      param.relu_enable = do_relu ? 1 : 0;
      param.ps32_mode = 0;
      param.w_is_const = 0;
      param.layer_id = layer_id;
      param.ins_val = 0; // symmetric quantization
      param.ins_fp =
          CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization
      CV18xx::tiu_convolution(&param);
    } else {
      cvk_tl_t tl_ifmap_oc_pos;
      uint32_t ifmap_count = align_up(ih * iw, CV18xx::EU_BYTES);
      tl_ifmap_oc_pos.start_address = tl_ifmap_step.start_address +
                                      (oc_pos / CV18xx::NPU_NUM) * ifmap_count;
      tl_ifmap_oc_pos.fmt = CVK_FMT_I8;
      tl_ifmap_oc_pos.shape = CV18xx::tl_shape_t4(n, cur_oc, ih, iw);
      tl_ifmap_oc_pos.stride = CV18xx::tl_default_stride(
          tl_ifmap_oc_pos.shape, CVK_FMT_I8, /*eu_align=*/1);
      tl_ifmap_oc_pos.stride.n =
          tl_ifmap_oc_pos.stride.c * ceiling_func(oc, CV18xx::NPU_NUM);
      // tl_filter shape for tiu != shape for load
      tl_filter.shape = CV18xx::tl_shape_t4(1, cur_oc, kh, kw);
      tl_filter.stride =
          CV18xx::tl_default_stride(tl_filter.shape, CVK_FMT_I8, /*eu_aign*/ 1);
      cvk_tiu_depthwise_convolution_param_t param = {0};
      param.ofmap = &tl_ofmap_oc_pos;
      param.ifmap = &tl_ifmap_oc_pos;
      param.weight = &tl_filter;
      param.chl_quan_param = &tl_perchannel_oc_pos;
      param.ins_h = ins_h;
      param.ins_w = ins_w;
      param.ins_last_h = 0;
      param.ins_last_w = 0;
      param.pad_top = pad_t;
      param.pad_bottom = pad_b;
      param.pad_left = pad_l;
      param.pad_right = pad_r;
      param.stride_h = sh;
      param.stride_w = sw;
      param.dilation_h = dh;
      param.dilation_w = dw;
      param.has_bias = with_bias ? 1 : 0;
      param.relu_enable = do_relu ? 1 : 0;
      param.layer_id = layer_id;
      param.ins_val = 0; // symmetric quantization
      param.ins_fp =
          CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization
      CV18xx::tiu_depthwise_convolution(&param);
    }
    if (do_leaky_relu) {
      // reuse the filter as working mem
      assert(oc_step == (uint32_t)CV18xx::NPU_NUM);
      assert(g == 1);
      // assert(n == 1);
      // work space is kh * kw * ic / g
      // need 2 work space, and align down to CV18xx::EU_BYTES
      // workload is oh * ow
      uint32_t relu_step =
          (kh * kw * ic / g / CV18xx::EU_BYTES) * CV18xx::EU_BYTES;
      uint32_t relu_size = oh * ow;
      uint32_t la_relu_working = tl_filter.start_address;
      for (int n_step = 0; n_step < (int)n; n_step++) {
        uint32_t relu_pos = 0;
        while (relu_pos < relu_size) {
          uint32_t relu_cur = std::min(relu_size - relu_pos, relu_step);
          // both input and output of the leaky relu
          cvk_tl_t tl_relu;
          tl_relu.start_address = tl_ofmap_oc_pos.start_address + relu_pos +
                                  n_step * tl_ofmap_oc_pos.stride.n;
          tl_relu.fmt = CVK_FMT_I8;
          tl_relu.shape = CV18xx::tl_shape_t4(1, cur_oc, 1, relu_cur);
          tl_relu.stride = CV18xx::tl_default_stride(tl_relu.shape, CVK_FMT_I8,
                                                     /*eu_align=*/1);
          // working has same shape as relu, different address
          cvk_tl_t tl_relu_working = tl_relu;
          tl_relu_working.start_address = la_relu_working;
          tl_leaky_relu(layer_id, tl_relu, tl_relu_working, rshift_pos,
                        m_i8_pos, rshift_neg, m_i8_neg);
          relu_pos += relu_cur;
        }
      }
    }
    if (do_store) {
      // because this store happens the same time as the above tiu operation
      // we can't store the data for this flip, we store the data of the
      // opposite flip
      if (first) {
        first = false;
      } else {
        cvk_tl_t tl_ofmap_oc_pos_flip_back;
        // uint32_t ofmap_count = align_up(oh * ow, CV18xx::EU_BYTES);
        tl_ofmap_oc_pos_flip_back.start_address =
            tl_ofmap.start_address +
            (oc_pos_flip_back / CV18xx::NPU_NUM) * ofmap_count;
        tl_ofmap_oc_pos_flip_back.fmt = CVK_FMT_I8;
        tl_ofmap_oc_pos_flip_back.shape =
            CV18xx::tl_shape_t4(n, cur_oc_flip_back, oh, ow);
        tl_ofmap_oc_pos_flip_back.stride = CV18xx::tl_default_stride(
            tl_ofmap_oc_pos_flip_back.shape, CVK_FMT_I8, /*eu_align=*/1);
        tl_ofmap_oc_pos_flip_back.stride.n =
            tl_ofmap_oc_pos_flip_back.stride.c *
            ceiling_func(oc, CV18xx::NPU_NUM);
        gaddr_t ga_ofmap_oc_pos_flip_back =
            ga_ofmap + oc_pos_flip_back * oh * ow;
        cvk_tg_stride_t ofmap_gstride = {oc * oh * ow, oh * ow, ow};
        CV18xx::parallel_disable();
        CV18xx::tdma_store_stride(&tl_ofmap_oc_pos_flip_back,
                                  ga_ofmap_oc_pos_flip_back, ofmap_gstride);
        CV18xx::parallel_enable();
      }
      oc_pos_flip_back = oc_pos;
      cur_oc_flip_back = cur_oc;
    }
    flip = 1 - flip;
  }

  if (do_store) {
    // do the last flip
    cvk_tl_t tl_ofmap_oc_pos_flip_back;
    uint32_t ofmap_count = align_up(oh * ow, CV18xx::EU_BYTES);
    tl_ofmap_oc_pos_flip_back.start_address =
        tl_ofmap.start_address +
        (oc_pos_flip_back / CV18xx::NPU_NUM) * ofmap_count;
    tl_ofmap_oc_pos_flip_back.fmt = CVK_FMT_I8;
    tl_ofmap_oc_pos_flip_back.shape =
        CV18xx::tl_shape_t4(n, cur_oc_flip_back, oh, ow);
    tl_ofmap_oc_pos_flip_back.stride = CV18xx::tl_default_stride(
        tl_ofmap_oc_pos_flip_back.shape, CVK_FMT_I8, /*eu_align=*/1);
    tl_ofmap_oc_pos_flip_back.stride.n =
        tl_ofmap_oc_pos_flip_back.stride.c * ceiling_func(oc, CV18xx::NPU_NUM);

    gaddr_t ga_ofmap_oc_pos_flip_back = ga_ofmap + oc_pos_flip_back * oh * ow;

    cvk_tg_stride_t ofmap_gstride = {oc * oh * ow, oh * ow, ow};
    CV18xx::parallel_disable();
    CV18xx::tdma_store_stride(&tl_ofmap_oc_pos_flip_back,
                              ga_ofmap_oc_pos_flip_back, ofmap_gstride);
    CV18xx::parallel_enable();
  }
}

void cvi_backend_tl_conv_LA(
    uint32_t layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap, gaddr_t ga_filter,
    gaddr_t ga_perchannel, uint32_t input_n, uint32_t input_c, uint32_t input_h,
    uint32_t input_w, uint32_t groups, uint32_t output_c, uint32_t output_h,
    uint32_t output_w, uint16_t kh, uint16_t kw, uint8_t dilation_h,
    uint8_t dilation_w, uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left,
    uint8_t pad_right, uint8_t stride_h, uint8_t stride_w, uint8_t insert_h,
    uint8_t insert_w, bool result_add, bool with_bias, bool do_relu,
    bool do_ic_alignment) {

  conv_args_t args;
  args.n = input_n;
  args.ic = input_c;
  args.ih = input_h;
  args.iw = input_w;
  args.g = groups;
  args.oc = output_c;
  args.oh = output_h;
  args.ow = output_w;
  args.kh = kh;
  args.kw = kw;
  args.dh = dilation_h;
  args.dw = dilation_w;
  args.pad_t = pad_top;
  args.pad_b = pad_bottom;
  args.pad_l = pad_left;
  args.pad_r = pad_right;
  args.sh = stride_h;
  args.sw = stride_w;
  args.ins_h = insert_h;
  args.ins_w = insert_w;
  args.result_add = false;
  args.with_bias = with_bias;
  args.do_relu = do_relu;
  CV18xx::set_layer_id(layer_id);
  if (do_ic_alignment && (input_c % 2 != 0)) {
    assert(input_c > 1);
    args.ic = input_c + 1;
  }

  conv_la_oc_step(layer_id, ga_ifmap, ga_ofmap, ga_filter, ga_perchannel, args,
                  CV18xx::NPU_NUM);
}

void cvi_backend_tl_conv_LW(
    uint32_t layer_id, laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_working,
    gaddr_t ga_filter, gaddr_t ga_perchannel, uint32_t input_n,
    uint32_t input_c, uint32_t input_h, uint32_t input_w, uint32_t groups,
    uint32_t output_c, uint32_t output_h, uint32_t output_w, uint16_t kh,
    uint16_t kw, uint8_t dilation_h, uint8_t dilation_w, uint8_t pad_top,
    uint8_t pad_bottom, uint8_t pad_left, uint8_t pad_right, uint8_t stride_h,
    uint8_t stride_w, uint8_t insert_h, uint8_t insert_w, bool result_add,
    bool with_bias, bool do_relu, bool do_store, gaddr_t ga_ofmap,
    bool do_leaky_relu, int8_t rshift_pos, int8_t m_i8_pos, int8_t rshift_neg,
    int8_t m_i8_neg, bool do_ic_alignment, bool compressed_weight) {

  conv_args_t args;
  args.n = input_n;
  args.ic = input_c;
  args.ih = input_h;
  args.iw = input_w;
  args.g = groups;
  args.oc = output_c;
  args.oh = output_h;
  args.ow = output_w;
  args.kh = kh;
  args.kw = kw;
  args.dh = dilation_h;
  args.dw = dilation_w;
  args.pad_t = pad_top;
  args.pad_b = pad_bottom;
  args.pad_l = pad_left;
  args.pad_r = pad_right;
  args.sh = stride_h;
  args.sw = stride_w;
  args.ins_h = insert_h;
  args.ins_w = insert_w;
  args.result_add = false;
  args.with_bias = with_bias;
  args.do_relu = do_relu;
  CV18xx::set_layer_id(layer_id);
  if (do_ic_alignment && (input_c % 2 != 0)) {
    assert(input_c > 1);
    args.ic = input_c + 1;
  }

  conv_lw_oc_step(layer_id, la_ifmap, la_ofmap, la_working, ga_filter,
                  ga_perchannel, args, CV18xx::NPU_NUM, do_store, ga_ofmap,
                  do_leaky_relu, rshift_pos, m_i8_pos, rshift_neg, m_i8_neg,
                  compressed_weight);
}

void cvi_backend_tl_conv(
    uint32_t layer_id, laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_weight,
    laddr_t la_working, laddr_t la_perchannel, int input_n, int input_c,
    int input_h, int input_w, int group, int output_c, int output_h,
    int output_w, uint32_t kh, uint32_t kw, uint32_t dilation_h,
    uint32_t dilation_w, uint32_t pad_h_top, uint32_t pad_h_bottom,
    uint32_t pad_w_left, uint32_t pad_w_right, uint32_t stride_h,
    uint32_t stride_w, uint32_t insert_h, uint32_t insert_w,
    uint32_t result_add, uint32_t ctrl, bool do_bias, bool do_relu, float slope,
    int rshift, int rshift_len, int8_t rshift_pos, int8_t rshift_neg,
    int8_t m_i8_pos, int8_t m_i8_neg, bool do_ic_alignment) {

  LLVM_DEBUG(
      llvm::errs() << llvm::format(
          "cvi_backend_tl_conv:\n"
          "    layer_id %d\n"
          "    la_ifmap 0x%lx, la_ofmap_0x%lx, la_weight 0x%lx, la_perchannel "
          "0x%lx\n"
          "    la_working 0x%lx\n"
          "    in(%d, %d, %d, %d), out(,%d, %d, %d), group %d kernel(%d,%d)\n"
          "    pad(%d, %d, %d, %d), stride(%d, %d), result_add %d\n"
          "    ctrl 0x%x, rshift %d, do_bias %d, do_relu %d\n"
          "    rshift_len %d\n",
          layer_id, la_ifmap, la_ofmap, la_weight, la_perchannel, la_working,
          input_n, input_c, input_h, input_w, output_c, output_h, output_w,
          group, kh, kw, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right,
          stride_h, stride_w, result_add, ctrl, rshift, do_bias, do_relu,
          rshift_len));
  if (do_ic_alignment && (input_c % 2 != 0)) {
    input_c = input_c + 1;
  }
  // input
  cvk_tl_shape_t tl_input_shape =
      CV18xx::tl_shape_t4(input_n, input_c, input_h, input_w);
  cvk_tl_t tl_input;
  tl_input.start_address = la_ifmap;
  tl_input.fmt = CVK_FMT_I8;
  tl_input.shape = tl_input_shape;
  tl_input.stride = CV18xx::tl_default_stride(tl_input_shape, CVK_FMT_I8, 1);

  // output
  cvk_tl_shape_t tl_output_shape =
      CV18xx::tl_shape_t4(input_n, output_c, output_h, output_w);
  cvk_tl_t tl_output;
  tl_output.start_address = la_ofmap;
  tl_output.fmt = CVK_FMT_I8;
  tl_output.shape = tl_output_shape;
  tl_output.stride = CV18xx::tl_default_stride(tl_output_shape, CVK_FMT_I8, 1);

  // weight
  cvk_tl_t tl_weight;
  if (group == input_c && group == output_c && group != 1) {
    cvk_tl_shape_t tl_weight_shape = CV18xx::tl_shape_t4(1, output_c, kh, kw);
    tl_weight.start_address = la_weight;
    tl_weight.fmt = CVK_FMT_I8;
    tl_weight.shape = tl_weight_shape;
    tl_weight.stride =
        CV18xx::tl_default_stride(tl_weight_shape, CVK_FMT_I8, 1);
  } else {
    cvk_tl_shape_t tl_weight_shape =
        CV18xx::tl_shape_t4(input_c, output_c, kh, kw);
    tl_weight.start_address = la_weight;
    tl_weight.fmt = CVK_FMT_I8;
    tl_weight.shape = tl_weight_shape;
    tl_weight.stride =
        CV18xx::tl_default_stride(tl_weight_shape, CVK_FMT_I8, 0);
  }

  bool do_chl_quan = rshift_len ? true : false;
  // only support chl_quan now.
  assert(do_chl_quan == true);
  cvk_tl_t tl_chl_quan_param = {0};

  // Per-channel quantization
  tl_chl_quan_param.start_address = la_perchannel;
  tl_chl_quan_param.fmt = CVK_FMT_I8;
  tl_chl_quan_param.shape = CV18xx::tl_shape_t4(1, output_c, 1, 1);
  tl_chl_quan_param.stride =
      CV18xx::tl_default_stride(tl_chl_quan_param.shape, CVK_FMT_I8, 0);

  bool do_conv_relu = do_relu;
  bool do_leaky_relu = false;

  if (slope != 0.0) {
    do_leaky_relu = true;
    do_conv_relu = false;
  }

  LLVM_DEBUG(llvm::errs() << "do_leaky_relu/do_conv_relu:" << do_leaky_relu
                          << "/" << do_conv_relu << "\n";);

  if (group == 1) {
    auto oc_step = output_c;
    if (output_c > MAX_TIU_CHL) {
      int i = 2;
      do {
        oc_step = align_up(ceiling_func(output_c, i++),
                           CV18xx::NPU_NUM * CV18xx::EU_BYTES);
      } while (oc_step > MAX_TIU_CHL);

      llvm::errs() << "output_c(" << output_c << ") is larger than "
                   << MAX_TIU_CHL << ", need to split it with step " << oc_step
                   << "\n";
    }
    for (int32_t oc_pos = 0; oc_pos < output_c; oc_pos += oc_step) {
      auto cur_oc = std::min(oc_step, output_c - oc_pos);
      cvk_tl_t output, weight, perchannel;
      cvk_tl_shape_t output_shape =
          CV18xx::tl_shape_t4(input_n, cur_oc, output_h, output_w);
      output.start_address =
          la_ofmap + (oc_pos / CV18xx::NPU_NUM) * tl_output.stride.c;
      output.fmt = CVK_FMT_I8;
      output.shape = output_shape;
      output.stride = tl_output.stride;

      // filter shape for tiu != shape for load
      cvk_tl_shape_t weight_shape =
          CV18xx::tl_shape_t4(input_c, cur_oc, kh, kw);
      weight.start_address =
          la_weight + (oc_pos / CV18xx::NPU_NUM) * kh * kw * input_c;
      weight.fmt = CVK_FMT_I8;
      weight.shape = weight_shape;
      weight.stride = CV18xx::tl_default_stride(weight_shape, CVK_FMT_I8, 0);

      perchannel.start_address =
          la_perchannel +
          (oc_pos / CV18xx::NPU_NUM) * CV18xx::chan_quan_param_size(do_bias);
      perchannel.fmt = CVK_FMT_I8;
      perchannel.shape = {1, static_cast<uint32_t>(cur_oc), 1, 1};
      perchannel.stride =
          CV18xx::tl_default_stride(perchannel.shape, CVK_FMT_I8, 0);

      // Per-channel quantization
      cvk_tiu_convolution_param_t param = {nullptr};
      param.ofmap = &output;
      param.ifmap = &tl_input;
      param.weight = &weight;
      param.chl_quan_param = &perchannel;
      param.ins_h = insert_h;
      param.ins_w = insert_w;
      param.ins_last_h = 0;
      param.ins_last_w = 0;
      param.pad_top = pad_h_top;
      param.pad_bottom = pad_h_bottom;
      param.pad_left = pad_w_left;
      param.pad_right = pad_w_right;
      param.stride_h = stride_h;
      param.stride_w = stride_w;
      param.dilation_h = dilation_h;
      param.dilation_w = dilation_w;
      param.has_bias = do_bias ? 1 : 0;
      param.relu_enable = do_conv_relu;
      param.ps32_mode = 0;
      param.w_is_const = 0;
      param.layer_id = layer_id;
      param.ins_val = 0; // symmetric quantization
      param.ins_fp =
          CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization
      CV18xx::tiu_convolution(&param);
    }

    if (do_leaky_relu) {
      cvk_tl_t tl_relu_working = tl_output;
      tl_relu_working.start_address = la_working;
      tl_leaky_relu(layer_id, tl_output, tl_relu_working, rshift_pos, m_i8_pos,
                    rshift_neg, m_i8_neg);
    }
  } else if (group == input_c && group == output_c) {
    // depthwise convolution
    // Per-channel quantization
    cvk_tiu_depthwise_convolution_param_t param = {0};
    param.ofmap = &tl_output;
    param.ifmap = &tl_input;
    param.weight = &tl_weight;
    param.chl_quan_param = &tl_chl_quan_param;
    param.ins_h = insert_h;
    param.ins_w = insert_w;
    param.ins_last_h = 0;
    param.ins_last_w = 0;
    param.pad_top = pad_h_top;
    param.pad_bottom = pad_h_bottom;
    param.pad_left = pad_w_left;
    param.pad_right = pad_w_right;
    param.stride_h = stride_h;
    param.stride_w = stride_w;
    param.dilation_h = dilation_h;
    param.dilation_w = dilation_w;
    param.has_bias = do_bias ? 1 : 0;
    param.relu_enable = do_conv_relu;
    param.layer_id = layer_id;
    param.ins_val = 0;                                // symmetric quantization
    param.ins_fp = CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization
    CV18xx::tiu_depthwise_convolution(&param);

    if (do_leaky_relu) {
      cvk_tl_t tl_relu_working = tl_output;
      tl_relu_working.start_address = la_working;
      tl_leaky_relu(layer_id, tl_output, tl_relu_working, rshift_pos, m_i8_pos,
                    rshift_neg, m_i8_neg);
    }
  } else {
    // Convolution
    int ic = input_c / group;
    int oc = output_c / group;
    int bottomc_per_NPU = ceiling_func(input_c, CV18xx::NPU_NUM);
    int topc_per_NPU = ceiling_func(output_c, CV18xx::NPU_NUM);
    int bottom_csize_local = ALIGN(input_h * input_w, CV18xx::EU_BYTES);
    int top_csize_local = ALIGN(output_h * output_w, CV18xx::EU_BYTES);
    //
    int bias_count = ceiling_func(oc, CV18xx::NPU_NUM);
    int bias_usize = CV18xx::chan_quan_param_size(do_bias);
    int bias_size = bias_count * bias_usize;

    for (int ig = 0; ig < group; ig++) {
      int bottom_start_npu_idx = (ig * ic) % CV18xx::NPU_NUM;
      int top_start_npu_idx = (ig * oc) % CV18xx::NPU_NUM;

      for (int nidx = 0; nidx < input_n; nidx++) {
        uint32_t top_local_shift =
            (nidx * topc_per_NPU + (ig * oc) / CV18xx::NPU_NUM) *
            top_csize_local;
        uint32_t bottom_local_shift =
            (nidx * bottomc_per_NPU + (ig * ic) / CV18xx::NPU_NUM) *
            bottom_csize_local;
        uint32_t bottom_addr = bottom_start_npu_idx * CV18xx::LMEM_BYTES +
                               tl_input.start_address + bottom_local_shift;
        uint32_t top_addr = top_start_npu_idx * CV18xx::LMEM_BYTES +
                            tl_output.start_address + top_local_shift;
        uint32_t weight_addr = top_start_npu_idx * CV18xx::LMEM_BYTES +
                               ((ig * oc) / CV18xx::NPU_NUM) * ic * kh * kw +
                               tl_weight.start_address;

        // not need to add top_start_npu_idx for bias address,
        // since opd2_addr has only 16 bits
        // here we add the top_start_npu_idx only for good code review
        // bias will use result opd's top_start_npu_idx as hw default
        uint32_t bias_local_shift =
            (ig * oc / CV18xx::NPU_NUM) * ALIGN(bias_size, CV18xx::EU_BYTES);
        uint32_t perchannel_addr = top_start_npu_idx * CV18xx::LMEM_BYTES +
                                   la_perchannel + bias_local_shift;

        // input
        cvk_tl_shape_t input_shape =
            CV18xx::tl_shape_t4(1, ic, input_h, input_w);
        cvk_tl_t input;
        input.start_address = bottom_addr;
        input.fmt = CVK_FMT_I8;
        input.shape = input_shape;
        input.stride =
            CV18xx::tl_default_stride(input_shape, CVK_FMT_I8, 1); // EU-aligned

        // weight
        cvk_tl_shape_t weight_shape = CV18xx::tl_shape_t4(ic, oc, kh, kw);
        cvk_tl_t weight;
        weight.start_address = weight_addr;
        weight.fmt = CVK_FMT_I8;
        weight.shape = weight_shape;
        weight.stride = CV18xx::tl_default_stride(weight_shape, CVK_FMT_I8,
                                                  0); // Not EU-aligned

        // output
        cvk_tl_shape_t output_shape =
            CV18xx::tl_shape_t4(1, oc, output_h, output_w);
        cvk_tl_t output;
        output.start_address = top_addr;
        output.fmt = CVK_FMT_I8;
        output.shape = output_shape;
        output.stride = CV18xx::tl_default_stride(output_shape, CVK_FMT_I8,
                                                  1); // EU-aligned

        // Per-channel quantization
        tl_chl_quan_param.start_address = perchannel_addr;
        tl_chl_quan_param.shape = CV18xx::tl_shape_t4(1, oc, 1, 1);
        tl_chl_quan_param.stride =
            CV18xx::tl_default_stride(tl_chl_quan_param.shape, CVK_FMT_I8, 0);

        cvk_tiu_convolution_param_t param = {nullptr};
        param.ofmap = &output;
        param.ifmap = &input;
        param.weight = &weight;
        param.chl_quan_param = &tl_chl_quan_param;
        param.ins_h = insert_h;
        param.ins_w = insert_w;
        param.ins_last_h = 0;
        param.ins_last_w = 0;
        param.pad_top = pad_h_top;
        param.pad_bottom = pad_h_bottom;
        param.pad_left = pad_w_left;
        param.pad_right = pad_w_right;
        param.stride_h = stride_h;
        param.stride_w = stride_w;
        param.dilation_h = dilation_h;
        param.dilation_w = dilation_w;
        param.has_bias = do_bias ? 1 : 0;
        param.relu_enable = do_conv_relu;
        param.ps32_mode = 0;
        param.w_is_const = 0;
        param.layer_id = layer_id;
        param.ins_val = 0; // symmetric quantization
        param.ins_fp =
            CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization
        CV18xx::tiu_convolution(&param);

        if (do_leaky_relu) {
          cvk_tl_t tl_relu_working = tl_output;
          tl_relu_working.start_address = la_working;
          tl_leaky_relu(layer_id, tl_output, tl_relu_working, rshift_pos,
                        m_i8_pos, rshift_neg, m_i8_neg);
        }
      }
    }
  }
}

void cvi_backend_bf16_tl_conv(
    uint32_t layer_id, laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_weight,
    laddr_t la_working, laddr_t la_bias, int input_n, int input_c, int input_h,
    int input_w, int group, int output_c, int output_h, int output_w,
    uint32_t kh, uint32_t kw, uint32_t dilation_h, uint32_t dilation_w,
    uint32_t pad_h_top, uint32_t pad_h_bottom, uint32_t pad_w_left,
    uint32_t pad_w_right, uint32_t stride_h, uint32_t stride_w,
    uint32_t insert_h, uint32_t insert_w, bool with_bias, bool do_relu) {

  LLVM_DEBUG(
      llvm::errs() << llvm::format(
          "cvi_backend_bf16_tl_conv:\n"
          "    layer_id %d\n"
          "    la_ifmap 0x%lx, la_ofmap_0x%lx, la_weight 0x%lx, la_bias 0x%lx\n"
          "    la_working 0x%lx\n"
          "    in(%d, %d, %d, %d), out(,%d, %d, %d), group %d kernel(%d,%d)\n"
          "    pad(%d, %d, %d, %d), stride(%d, %d)\n"
          "    do_bias %d, do_relu %d\n"
          "    \n",
          layer_id, la_ifmap, la_ofmap, la_weight, la_bias, la_working, input_n,
          input_c, input_h, input_w, output_c, output_h, output_w, group, kh,
          kw, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, stride_h,
          stride_w, with_bias, do_relu));

  // input
  cvk_tl_shape_t tl_input_shape =
      CV18xx::tl_shape_t4(input_n, input_c, input_h, input_w);
  cvk_tl_t tl_input;
  tl_input.start_address = la_ifmap;
  tl_input.fmt = CVK_FMT_BF16;
  tl_input.shape = tl_input_shape;
  tl_input.stride = CV18xx::tl_default_stride(tl_input_shape, CVK_FMT_BF16, 1);

  // output
  cvk_tl_shape_t tl_output_shape =
      CV18xx::tl_shape_t4(input_n, output_c, output_h, output_w);
  cvk_tl_t tl_output;
  tl_output.start_address = la_ofmap;
  tl_output.fmt = CVK_FMT_BF16;
  tl_output.shape = tl_output_shape;
  tl_output.stride =
      CV18xx::tl_default_stride(tl_output_shape, CVK_FMT_BF16, 1);

  // weight
  cvk_tl_t tl_weight;
  if (group == input_c && group == output_c && group != 1) {
    cvk_tl_shape_t tl_weight_shape = CV18xx::tl_shape_t4(1, output_c, kh, kw);
    tl_weight.start_address = la_weight;
    tl_weight.fmt = CVK_FMT_BF16;
    tl_weight.shape = tl_weight_shape;
    tl_weight.stride =
        CV18xx::tl_default_stride(tl_weight_shape, CVK_FMT_BF16, 1);
  } else {
    cvk_tl_shape_t tl_weight_shape =
        CV18xx::tl_shape_t4(input_c, output_c, kh, kw);
    tl_weight.start_address = la_weight;
    tl_weight.fmt = CVK_FMT_BF16;
    tl_weight.shape = tl_weight_shape;
    tl_weight.stride =
        CV18xx::tl_default_stride(tl_weight_shape, CVK_FMT_BF16, 0);
  }

  //
  cvk_tl_t tl_bias;
  tl_bias.start_address = la_bias;
  tl_bias.fmt = CVK_FMT_BF16;
  tl_bias.shape = CV18xx::tl_shape_t4(2, output_c, 1, 1);
  tl_bias.stride = CV18xx::tl_default_stride(tl_bias.shape, CVK_FMT_BF16, 0);

  if (group == 1) {
    cvk_tiu_pt_convolution_param_t param = {0};
    param.ofmap = &tl_output;
    param.ifmap = &tl_input;
    param.weight = &tl_weight;
    param.bias = with_bias ? &tl_bias : 0;
    param.ins_h = insert_h;
    param.ins_w = insert_w;
    param.ins_last_h = 0;
    param.ins_last_w = 0;
    param.pad_top = pad_h_top;
    param.pad_bottom = pad_h_bottom;
    param.pad_left = pad_w_left;
    param.pad_right = pad_w_right;
    param.stride_h = stride_h;
    param.stride_w = stride_w;
    param.dilation_h = dilation_h;
    param.dilation_w = dilation_w;
    param.relu_enable = do_relu;
    param.ps32_mode = 0;
    param.w_is_const = 0;
    param.layer_id = layer_id;
    param.ins_val = 0;                                // symmetric quantization
    param.ins_fp = CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization

    CV18xx::tiu_pt_convolution(&param);
  } else if (group == input_c && group == output_c) {
    // depthwise conv
    cvk_tiu_depthwise_pt_convolution_param_t param = {0};
    param.ofmap = &tl_output;
    param.ifmap = &tl_input;
    param.weight = &tl_weight;
    param.bias = with_bias ? &tl_bias : 0;
    param.ins_h = insert_h;
    param.ins_w = insert_w;
    param.ins_last_h = 0;
    param.ins_last_w = 0;
    param.pad_top = pad_h_top;
    param.pad_bottom = pad_h_bottom;
    param.pad_left = pad_w_left;
    param.pad_right = pad_w_right;
    param.stride_h = stride_h;
    param.stride_w = stride_w;
    param.dilation_h = dilation_h;
    param.dilation_w = dilation_w;
    param.rshift_bits = 0;
    param.relu_enable = do_relu;
    param.layer_id = layer_id;
    param.ins_val = 0;                                // symmetric quantization
    param.ins_fp = CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization
    CV18xx::tiu_pt_depthwise_convolution(&param);

  } else {
    // group conv
    int ic = input_c / group;
    int oc = output_c / group;
    int bf16_usize = 2;
    int bottomc_per_NPU = ceiling_func(input_c, CV18xx::NPU_NUM);
    int topc_per_NPU = ceiling_func(output_c, CV18xx::NPU_NUM);
    int bottom_csize_local =
        align_up(input_h * input_w * bf16_usize, CV18xx::EU_BYTES);
    int top_csize_local =
        align_up(output_h * output_w * bf16_usize, CV18xx::EU_BYTES);
    //
    int bias_count = ceiling_func(oc, CV18xx::NPU_NUM);
    int bias_size = bias_count * bf16_usize;

    for (int ig = 0; ig < group; ig++) {
      int bottom_start_npu_idx = (ig * ic) % CV18xx::NPU_NUM;
      int top_start_npu_idx = (ig * oc) % CV18xx::NPU_NUM;

      for (int nidx = 0; nidx < input_n; nidx++) {
        uint32_t top_local_shift =
            (nidx * topc_per_NPU + (ig * oc) / CV18xx::NPU_NUM) *
            top_csize_local;
        uint32_t bottom_local_shift =
            (nidx * bottomc_per_NPU + (ig * ic) / CV18xx::NPU_NUM) *
            bottom_csize_local;
        uint32_t bottom_addr = bottom_start_npu_idx * CV18xx::LMEM_BYTES +
                               tl_input.start_address + bottom_local_shift;
        uint32_t top_addr = top_start_npu_idx * CV18xx::LMEM_BYTES +
                            tl_output.start_address + top_local_shift;
        uint32_t weight_addr =
            top_start_npu_idx * CV18xx::LMEM_BYTES +
            ((ig * oc) / CV18xx::NPU_NUM) * ic * kh * kw * bf16_usize +
            tl_weight.start_address;

        // not need to add top_start_npu_idx for bias address,
        // since opd2_addr has only 16 bits
        // here we add the top_start_npu_idx only for good code review
        // bias will use result opd's top_start_npu_idx as hw default
        uint32_t bias_local_shift = (ig * oc / CV18xx::NPU_NUM) * bias_size;
        uint32_t bias_addr = la_bias + bias_local_shift;

        // input
        cvk_tl_shape_t input_shape = {1, static_cast<uint32_t>(ic),
                                      static_cast<uint32_t>(input_h),
                                      static_cast<uint32_t>(input_w)};
        cvk_tl_t input;
        input.start_address = bottom_addr;
        input.fmt = CVK_FMT_BF16;
        input.shape = input_shape;
        input.stride = CV18xx::tl_default_stride(input_shape, CVK_FMT_BF16, 1);

        // weight
        cvk_tl_shape_t weight_shape = {
            static_cast<uint32_t>(ic), static_cast<uint32_t>(oc),
            static_cast<uint32_t>(kh), static_cast<uint32_t>(kw)};
        cvk_tl_t weight;
        weight.start_address = weight_addr;
        weight.fmt = CVK_FMT_BF16;
        weight.shape = weight_shape;
        weight.stride =
            CV18xx::tl_default_stride(weight_shape, CVK_FMT_BF16, 0);
        // output
        cvk_tl_shape_t output_shape = {1, static_cast<uint32_t>(oc),
                                       static_cast<uint32_t>(output_h),
                                       static_cast<uint32_t>(output_w)};
        cvk_tl_t output;
        output.start_address = top_addr;
        output.fmt = CVK_FMT_BF16;
        output.shape = output_shape;
        output.stride =
            CV18xx::tl_default_stride(output_shape, CVK_FMT_BF16, 1);

        // bias
        tl_bias.start_address = bias_addr;
        tl_bias.shape = {2, static_cast<uint32_t>(oc), 1, 1};
        tl_bias.stride =
            CV18xx::tl_default_stride(tl_bias.shape, CVK_FMT_BF16, 0);

        cvk_tiu_pt_convolution_param_t param = {nullptr};
        param.ofmap = &output;
        param.ifmap = &input;
        param.weight = &weight;
        param.bias = with_bias ? &tl_bias : 0;
        param.ins_h = insert_h;
        param.ins_w = insert_w;
        param.ins_last_h = 0;
        param.ins_last_w = 0;
        param.pad_top = pad_h_top;
        param.pad_bottom = pad_h_bottom;
        param.pad_left = pad_w_left;
        param.pad_right = pad_w_right;
        param.stride_h = stride_h;
        param.stride_w = stride_w;
        param.dilation_h = dilation_h;
        param.dilation_w = dilation_w;
        param.relu_enable = do_relu;
        param.ps32_mode = 0;
        param.w_is_const = 0;
        param.layer_id = layer_id;
        param.ins_val = 0; // symmetric quantization
        param.ins_fp =
            CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization
        CV18xx::tiu_pt_convolution(&param);
      }
    }
  }
}
} // namespace backend
} // namespace tpu_mlir

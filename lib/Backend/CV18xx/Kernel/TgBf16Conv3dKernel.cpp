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

#define DEBUG_TYPE "cvi_backend_bf16_conv3d_kernel"

using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace backend {
static void loadBias(uint64_t ga_bias, cvk_tl_t *tl_bias_al, int oc_pos,
                     int oc) {
  cvk_fmt_t fmt = tl_bias_al->fmt;
  cvk_tg_shape_t gm_bias_shape = {tl_bias_al->shape.n, (uint32_t)oc,
                                  tl_bias_al->shape.h, tl_bias_al->shape.w};
  cvk_tg_stride_t gm_bias_stride =
      CV18xx::tg_default_stride(gm_bias_shape, fmt);
  gm_bias_shape.c = tl_bias_al->shape.c;
  cvk_tg_t gm_bias;
  CV18xx::gmem_init_tensor(&gm_bias, gm_bias_shape, fmt);
  gm_bias.stride = gm_bias_stride;
  gm_bias.base_reg_index = CV18xx::getTdmaBaseSelectIndexFromGaddr(ga_bias);
  gm_bias.start_address = ga_bias + oc_pos * gm_bias.stride.c;

  cvk_tdma_g2l_tensor_copy_param_t param = {0};
  param.src = &gm_bias;
  param.dst = tl_bias_al;
  CV18xx::tdma_g2l_tensor_copy(&param);
}

// Input (n, ic, id, ih, iw)
static void loadInput(uint32_t layer_id, int n, int ic, int id, int ih, int iw,
                      int idi, uint64_t ga_input, cvk_tl_t *tl_input_al) {
  // reshape (n, ic, id, ih, iw) => (n, ic, id, ih*iw)
  cvk_fmt_t fmt = tl_input_al->fmt;
  cvk_tl_shape_t tl_shape = tl_input_al->shape;
  cvk_tl_t tl_input;
  CV18xx::lmem_init_tensor(&tl_input, tl_shape, fmt, tl_input_al->eu_align);
  tl_input.start_address = tl_input_al->start_address;

  uint32_t ds = (fmt == CVK_FMT_BF16) ? 2 : 1;
  cvk_tg_shape_t gm_input_shape =
      CV18xx::tg_shape_t4(n, ic, tl_shape.h, tl_shape.w);
  cvk_tg_stride_t gm_input_stride = {ic * id * ih * iw * ds, id * ih * iw * ds,
                                     iw * ds, ds};

  if (idi >= 0 && idi < id) {
    cvk_tg_t gm_input;
    CV18xx::gmem_init_tensor(&gm_input, gm_input_shape, fmt);
    gm_input.base_reg_index = CV18xx::getTdmaBaseSelectIndexFromGaddr(ga_input);
    gm_input.start_address = ga_input + (ih * iw * ds) * idi;
    gm_input.stride = gm_input_stride;

    cvk_tdma_g2l_tensor_copy_param_t param = {0};
    param.src = &gm_input;
    param.dst = &tl_input;
    CV18xx::tdma_g2l_tensor_copy(&param);
  } else {
    // clean up
    cvk_tl_shape_t tl_pad_shape =
        CV18xx::tl_shape_t4(n, ic, ih, iw * tl_input_al->stride.w);
    cvk_tl_t tl_pad;
    CV18xx::lmem_init_tensor(&tl_pad, tl_pad_shape, CVK_FMT_I8, /*eu_align=*/1);
    tl_pad.start_address = tl_input_al->start_address;
    CV18xx::tiu_zeros(layer_id, &tl_pad);
  }
}

// TPU weight (kd, oc, kh*kw, ic)
static void loadWeight(int oc_pos, int oc_step, int oc, int ic, int kd, int kh,
                       int kw, uint64_t ga_weight, cvk_tl_t *tl_weight_al) {
  cvk_fmt_t fmt = tl_weight_al->fmt;
  cvk_tg_shape_t gm_weight_shape = CV18xx::tg_shape_t4(kd, oc, kh * kw, ic);
  cvk_tg_stride_t gm_weight_stride =
      CV18xx::tg_default_stride(gm_weight_shape, fmt);
  gm_weight_shape.c = oc_step;
  cvk_tg_t gm_weight;
  CV18xx::gmem_init_tensor(&gm_weight, gm_weight_shape, fmt);
  gm_weight.stride = gm_weight_stride;
  gm_weight.base_reg_index = CV18xx::getTdmaBaseSelectIndexFromGaddr(ga_weight);
  gm_weight.start_address = ga_weight + oc_pos * gm_weight_stride.c;

  cvk_tdma_g2l_tensor_copy_param_t param = {0};
  param.src = &gm_weight;
  param.dst = tl_weight_al;
  CV18xx::tdma_g2l_tensor_copy(&param);
}

void initWeightForTiu(cvk_tl_t *tl_weight, cvk_tl_t *tl_weight_al, int oc,
                      int ic, int kh, int kw, int kdi, cvk_fmt_t fmt) {
  *tl_weight = *tl_weight_al;
  tl_weight->shape.n = 1;
  tl_weight->start_address =
      tl_weight_al->start_address + tl_weight->stride.n * kdi;
}

static int get_ps32_mode(int kdi, int kd) {
  if (kd == 1)
    return 0;

  if (kdi == 0)
    return 2; // [1]: write
  else if (kdi == (kd - 1))
    return 1; // [0]: read

  return 3; // [1]: write, [0]: read
}

static void compute(int n, int ic, int kh, int kw, uint8_t pad_top,
                    uint8_t pad_bot, uint8_t pad_left, uint8_t pad_right,
                    int oc_pos, int oc_step, int oc, int oh, int ow,
                    uint8_t stride_h, uint8_t stride_w, int ps32_mode,
                    bool do_relu, cvk_tl_t *tl_input_al, cvk_tl_t *tl_weight_al,
                    cvk_tl_t *tl_bias_al, cvk_tl_t *tl_output_al) {
  auto fmt = tl_input_al->fmt;
  cvk_tl_t *tl_output = tl_output_al;
  cvk_tl_t *tl_input = tl_input_al;
  cvk_tl_shape_t tl_weight_shape = CV18xx::tl_shape_t4(ic, oc_step, kh, kw);
  cvk_tl_t tl_weight;
  CV18xx::lmem_init_tensor(&tl_weight, tl_weight_shape, fmt, /*eu_align=*/0);
  tl_weight.start_address = tl_weight_al->start_address;
  tl_weight.stride = tl_weight_al->stride;

  cvk_tl_t *tl_bias = nullptr;
  if (tl_bias_al) {
    tl_bias = tl_bias_al;
  }

  // ps32_mode == 0 for di = 1 case
  bool ps_write = ps32_mode == 1 || ps32_mode == 0;

  cvk_tiu_pt_convolution_param_t param = {0};
  param.ifmap = tl_input;
  param.ofmap = tl_output;
  param.weight = &tl_weight;
  param.bias = (tl_bias_al && ps_write == 1) ? tl_bias : NULL;
  param.pad_top = pad_top;
  param.pad_bottom = pad_bot;
  param.pad_left = pad_left;
  param.pad_right = pad_right;
  param.stride_h = stride_h;
  param.stride_w = stride_w;
  param.dilation_h = 1;
  param.dilation_w = 1;
  param.ps32_mode = ps32_mode;
  param.ins_val = 0;                                // symmetric quantization
  param.ins_fp = CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization
  param.relu_enable = do_relu && ps_write;
  CV18xx::tiu_pt_convolution(&param);
}

static void storeOutput(int oc_pos, int oc_step, int oc, int od, int oh, int ow,
                        int odi, uint64_t ga_res, cvk_tl_t *tl_res) {
  cvk_fmt_t fmt = tl_res->fmt;
  uint32_t ds = (fmt == CVK_FMT_BF16) ? 2 : 1;

  // Global memory shape (n, oc, od, oh, ow)
  cvk_tg_shape_t tg_res_shape = {tl_res->shape.n, tl_res->shape.c,
                                 tl_res->shape.h, tl_res->shape.w};
  cvk_tg_stride_t tg_stride = {oc * od * oh * ow * ds, od * oh * ow * ds,
                               ow * ds, ds};
  uint32_t od_stride = oh * ow * ds;

  cvk_tg_t gm_res = {0};
  CV18xx::gmem_init_tensor(&gm_res, tg_res_shape, fmt);
  gm_res.base_reg_index = CV18xx::getTdmaBaseSelectIndexFromGaddr(ga_res);
  gm_res.start_address = ga_res + od_stride * odi + oc_pos * tg_stride.c;
  gm_res.stride = tg_stride;

  cvk_tdma_l2g_tensor_copy_param_t param = {0};
  memset(&param, 0, sizeof(param));
  param.src = tl_res;
  param.dst = &gm_res;
  CV18xx::tdma_l2g_tensor_copy(&param);
}

typedef struct {
  // tiling used
  int n;
  int oc;
  int ic;
  int h;
  int w;
  int oc_step;
  int oh_step;
  int ow_step;
  int ih_step;
  int iw_step;
  // generate cmd used
  // int cur_oc;
  int cur_oh;
  int cur_ow;
  int cur_ih;
  int cur_iw;
  int ih_top;
  int ih_bot;
  int iw_right;
  int iw_left;
  int ph_top;
  int ph_bot;
  int pw_left;
  int pw_right;
  int oh_top;
  int oh_bot;
  int ow_left;
  int ow_right;
} SLICES;

static int conv_split(SLICES &slices, int input_n, int input_c, int input_d,
                      int input_h, int input_w, int output_c, int output_d,
                      int output_h, int output_w, uint16_t kd, uint16_t kh,
                      uint16_t kw, uint16_t dilation_d, uint16_t dilation_h,
                      uint16_t dilation_w, uint8_t pad_d0, uint8_t pad_d1,
                      uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left,
                      uint8_t pad_right, uint8_t stride_d, uint8_t stride_h,
                      uint8_t stride_w, bool do_bias, bool do_relu,
                      cvk_fmt_t fmt) {

  // TODO: support groups, tile oc
  // FIXME: not hardcode
  int groups = 1;
  // bool do_chl_quan = false;
  // int num_oc_step = 1;

  int ic = input_c / groups;
  int oc = output_c / groups;
  int kh_extent = dilation_h * (kh - 1) + 1;
  int kw_extent = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_extent) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_extent) / stride_w + 1;
  int ih = input_h;
  int iw = input_w;
  int n = input_n;

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "conv_split =>\n"
                 "  groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
                 "  kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
                 "  stride (%d, %d), dilation (%d, %d)\n",
                 groups, input_n, input_c, input_h, input_w, input_n, oc, oh,
                 ow, kh, kw, pad_top, pad_bottom, pad_left, pad_right, stride_h,
                 stride_w, dilation_h, dilation_w));

  slices.n = 1;
  // slices.oc = ceiling_func(oc, CV18xx::NPU_NUM); // lane parallelism
  slices.oc = 1; //ceiling_func(oc, CV18xx::NPU_NUM);  // lane parallelism
  slices.ic = 1;
  slices.h = (ih + (4095 - 32 - 1)) / (4095 - 32); // 12bit, max 4095-32(lanes)
  slices.w = (iw + (4095 - 32 - 1)) / (4095 - 32); // 12bit, max 4095-32(lanes)

  //
  // Slices may not be a good way to find size
  // We may try to increase or decrease width in aligned with 4, 8, 16 ...
  // or specific height/width (8, 8), (16, 16) ...
  //
  // Split ow
  for (slices.w = 1; slices.w <= ow; ++slices.w) {
    int ow_step = ceiling_func(ow, slices.w);
    int iw_step = std::min((ow_step - 1) * stride_w + kw_extent, iw);

    if ((slices.w == 1) && (stride_w > 1)) {
      // For better DMA transfer efficiency, use whole width.
      //   E.g.
      //     ifmap (1, 512, 28, 28), kernel (1, 1), stride 2
      //
      //     input (27, 27) needed, but (27, 28) is better
      iw_step = std::min(iw_step + stride_w - 1, iw);
      slices.iw_step = iw_step;
    }

    // Split oh
    for (slices.h = 1; slices.h <= oh; ++slices.h) {
      for (slices.oc = 1; slices.oc <= oc; ++slices.oc) {
        int oc_step = ceiling_func(oc, slices.oc);

        // We may need to put EU-alignment info in one place
        cvk_tl_shape_t coeff_shape =
            CV18xx::tl_shape_t4(2, oc_step, 1, CV18xx::bytesize_of_fmt(fmt));

        uint32_t coeff_oc_step_size = 0;

        if (do_bias) {
          // bf16 take 16*2 bit
          assert(fmt == CVK_FMT_BF16);
          coeff_oc_step_size +=
              CV18xx::lmem_tensor_to_size(coeff_shape, fmt, /*eu_align=*/0);
        }

        // Add weight size
        coeff_oc_step_size += CV18xx::lmem_tensor_to_size(
            CV18xx::tl_shape_t4(ic, oc_step, kh, kw), fmt, /*eu_align=*/0);

        // split n
        for (slices.n = 1; slices.n <= n; ++slices.n) {
          int n_step = ceiling_func(n, slices.n);

          int oh_step = ceiling_func(oh, slices.h);
          int ih_step = std::min((oh_step - 1) * stride_h + kh_extent, ih);

          uint32_t total_needed = 0;

          uint32_t ofmap_size = CV18xx::lmem_tensor_to_size(
              CV18xx::tl_shape_t4(n_step * 2, oc_step, oh_step, ow_step), fmt,
              /*eu_align=*/1);
          total_needed += ofmap_size;

          uint32_t ifmap_size = CV18xx::lmem_tensor_to_size(
              CV18xx::tl_shape_t4(n_step, ic, ih_step, iw_step), fmt,
              /*eu_align=*/1);
          total_needed += ifmap_size;

          total_needed += coeff_oc_step_size;

          // TODO: support double buffer
          // Double buffers so that TDMA load and store can run during TIU
          // executes. total_needed *= 2;

          // Both prelu and leaky relu need tl_neg, tl_relu.
          // tl_relu, tl_neg are not from tmda and not final output.
          // One copy is enough.

          if (total_needed < (uint32_t)CV18xx::LMEM_BYTES) {
            // I try to maximize the local memory utilization,
            // but it causes large write latency, especially in cross-layer.
            // However TDMA engine can handle small data transfer efficiently.
            //
            // E.g. Resnet50 scale2b_branch2c in DDR3 platform.
            //   (1, 96, 56, 56) tiu 19471, store 31056, 77 fps
            //   (1, 32, 56, 56) tiu 6535, store 10376, 84 fps
            //
            // The load/store reorder may be useful in intra-layer and
            // inter-layer.
            //
            // The next-generation chip will do DMA store once intermediate
            // result is generated.
            //
            // The following is temporary solution.
            // I decrese the output channel size to trigger frequent DMA store.
            // So local memory is wasted.

            // DMA efficiency: OH * OW >= 256B
            if (1) {
              const uint32_t dma_min_size = 256;
              uint32_t ofmap_plane_size = oh_step * ow_step;

              if ((oc_step > CV18xx::NPU_NUM) &&
                  (ofmap_plane_size > (1 * dma_min_size))) {
                continue;
              }
              if ((oc_step > (2 * CV18xx::NPU_NUM)) &&
                  (ofmap_plane_size < dma_min_size)) {
                // even oh*ow is smaller, use at most 2xlanes_num
                continue;
              }
            }

            slices.oc_step = oc_step;
            slices.oh_step = oh_step;
            slices.ow_step = ow_step;
            slices.ih_step = ih_step;
            slices.iw_step = iw_step;

            LLVM_DEBUG(llvm::errs() << llvm::format(
                           "  Slices(n=%d, oc_step=%d, h=%d, w=%d), n_step %d, "
                           "ow_step %d, "
                           "iw_step %d, oh_step %d, ih_step %d"
                           ", coeff_oc_step_size %d, total_needed %d\n"
                           "  weight shape (%d, %d, %d, %d)\n"
                           "  ifmap shape (%d, %d, %d, %d)\n"
                           "  ofmap shape (%d, %d, %d, %d)\n",
                           slices.n, slices.oc_step, slices.h, slices.w, n_step,
                           ow_step, iw_step, oh_step, ih_step,
                           coeff_oc_step_size, total_needed, ic, oc_step, kh,
                           kw, n_step, ic, ih_step, iw_step, n_step, oc_step,
                           oh_step, ow_step));
            LLVM_DEBUG(llvm::errs() << "<= conv_split succeed\n");
            return total_needed;
          }

        } // for (slices.n = 1; slices.n < n; ++slices.n)
      }   // for (slices.oc = 1; slices.oc <= oc; ++slices.oc)

    } // for (slices.h = 1; slices.h <= oh; ++slices.h)

  } // for (slices.w = 1; slices.w <= ow; ++slices.ow)

  LLVM_DEBUG(llvm::errs() << "conv_split fail\n";);

  return -1;
}

static void get_steps(SLICES &slices, int stride_h, int stride_w, int kh_ext,
                      int kw_ext, int input_h, int input_w, uint8_t pad_top,
                      uint8_t pad_bottom, uint8_t pad_left, uint8_t pad_right,
                      int oh_pos, int oh, int oh_step, int ow_pos, int ow,
                      int ow_step) {

  // split w
  slices.cur_oh = std::min(oh - oh_pos, oh_step);
  slices.oh_top = oh_pos;
  slices.oh_bot = slices.oh_top + slices.cur_oh;
  slices.ih_top = std::max(slices.oh_top * stride_h - pad_top, 0);
  slices.ih_bot =
      std::min((slices.oh_bot - 1) * stride_h + kh_ext - pad_top, input_h);
  slices.cur_ih = slices.ih_bot - slices.ih_top;

  slices.ph_top = 0;
  if (slices.ih_top == 0) {
    slices.ph_top = pad_top - slices.oh_top * stride_h;
  }

  slices.ph_bot = 0;
  if (slices.ih_bot == input_h) {
    slices.ph_bot = (slices.oh_bot - 1) * stride_h + kh_ext - pad_top - input_h;
  }

  // split w
  slices.cur_ow = std::min(ow - ow_pos, ow_step);
  slices.ow_left = ow_pos;
  slices.ow_right = slices.ow_left + slices.cur_ow;
  slices.iw_left = std::max(slices.ow_left * stride_w - pad_left, 0);
  slices.iw_right =
      std::min((slices.ow_right - 1) * stride_w + kw_ext - pad_left, input_w);
  slices.cur_iw = slices.iw_right - slices.iw_left;

  slices.pw_left = 0;
  if (slices.iw_left == 0) {
    slices.pw_left = pad_left - slices.ow_left * stride_w;
  }

  slices.pw_right = 0;
  if (slices.iw_right == input_w) {
    slices.pw_right =
        (slices.ow_right - 1) * stride_w + kw_ext - pad_left - input_w;
  }
}

void cvi_backend_tg_bf16_conv3d_kernel(
    uint32_t layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap, gaddr_t ga_weight,
    gaddr_t ga_bias, int input_n, int input_c, int input_d, int input_h,
    int input_w, int output_c, int output_d, int output_h, int output_w,
    uint16_t kd, uint16_t kh, uint16_t kw, uint16_t dilation_d,
    uint16_t dilation_h, uint16_t dilation_w, uint8_t pad_d0, uint8_t pad_d1,
    uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left, uint8_t pad_right,
    uint8_t stride_d, uint8_t stride_h, uint8_t stride_w, bool has_bias,
    bool do_relu) {

  assert(input_n == 1 && "Only support batch 1");
  assert(stride_d == 1 && "Only support stride_d 1");
  assert(dilation_h == 1 && dilation_w == 1 && "Only support dilation_h/w = 1");
  assert(dilation_d == 1 && "Only support dilation_d = 1");
  assert(pad_d0 < kd && pad_d1 < kd && "Only support pad < kernel in d axis");

  cvk_fmt_t fmt = CVK_FMT_BF16;

  // TODO: support group
  SLICES slices;
  int r = conv_split(slices, input_n, input_c, input_d, input_h, input_w,
                     output_c, output_d, output_h, output_w, kd, kh, kw,
                     dilation_d, dilation_h, dilation_w, pad_d0, pad_d1,
                     pad_top, pad_bottom, pad_left, pad_right, stride_d,
                     stride_h, stride_w, has_bias, do_relu, fmt);

  assert(r != -1 && "conv tiling fail");

  int oh_step = slices.oh_step;
  int ow_step = slices.ow_step;
  int ih_step = slices.ih_step;
  int iw_step = slices.iw_step;

  // TODO: not hardcode
  int oc_step = slices.oc_step;
  int n_step = input_n;
  int groups = 1;
  int ic = input_c / groups;
  assert(groups == 1);

  int kh_ext = dilation_h * (kh - 1) + 1;
  int kw_ext = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_ext) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_ext) / stride_w + 1;

  int unit = CV18xx::bytesize_of_fmt(fmt);

  cvk_tl_shape_t tl_output_shape =
      CV18xx::tl_shape_t4(input_n, oc_step, oh_step, ow_step);
  cvk_tl_shape_t tl_output_shape_32 =
      CV18xx::tl_shape_t4(input_n * 2, oc_step, oh_step, ow_step);
  cvk_tl_shape_t tl_input_shape =
      CV18xx::tl_shape_t4(input_n, input_c, ih_step, iw_step);
  cvk_tl_shape_t tl_weight_shape =
      CV18xx::tl_shape_t4(kd, oc_step, kh * kw, input_c);
  cvk_tl_shape_t tl_bias_shape = CV18xx::tl_shape_t4(2, oc_step, 1, 1);

  cvk_tl_t *tl_output = CV18xx::lmem_alloc_tensor(tl_output_shape_32, fmt,
                                                  /*eu_align=*/1);
  tl_output->shape = tl_output_shape;
  cvk_tl_t *tl_input = CV18xx::lmem_alloc_tensor(tl_input_shape, fmt,
                                                 /*eu_align=*/1);
  cvk_tl_t *tl_weight = CV18xx::lmem_alloc_tensor(tl_weight_shape, fmt,
                                                  /*eu_align=*/0);
  cvk_tl_t *tl_bias = nullptr;
  if (has_bias)
    tl_bias = CV18xx::lmem_alloc_tensor(tl_bias_shape, fmt, /*eu_align=*/0);

  assert(tl_output && tl_input && tl_weight && "Expect all allocated");

  for (int ig = 0; ig < groups; ++ig) {
    for (int oc_pos = 0; oc_pos < output_c; oc_pos += oc_step) {
      int cur_oc = std::min(output_c - oc_pos, oc_step);
      tl_weight->shape = CV18xx::tl_shape_t4(kd, cur_oc, kh * kw, input_c);
      loadWeight(oc_pos, cur_oc, output_c, input_c, kd, kh, kw, ga_weight,
                 tl_weight);
      if (has_bias) {
        tl_bias->shape.c = cur_oc;
        loadBias(ga_bias, tl_bias, oc_pos, output_c);
      }
      for (int n_pos = 0; n_pos < input_n; n_pos += n_step) {
        int cur_n = std::min(input_n - n_pos, n_step);
        // split h
        for (int oh_pos = 0; oh_pos < oh; oh_pos += oh_step) {
          // split w
          for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step) {

            SLICES s;
            get_steps(s, stride_h, stride_w, kh_ext, kw_ext, input_h, input_w,
                      pad_top, pad_bottom, pad_left, pad_right, oh_pos, oh,
                      oh_step, ow_pos, ow, ow_step);

            tl_output->shape =
                CV18xx::tl_shape_t4(cur_n, cur_oc, s.cur_oh, s.cur_ow);
            tl_output->stride =
                CV18xx::tl_default_stride(tl_output->shape, fmt, /*eu_aign=*/1);

            tl_input->shape =
                CV18xx::tl_shape_t4(cur_n, ic, s.cur_ih, s.cur_iw);
            tl_input->stride =
                CV18xx::tl_default_stride(tl_input->shape, fmt, /*eu_align=*/1);

            uint64_t ifmap_offset = unit * (s.ih_top * input_w + s.iw_left);
            uint64_t ofmap_offset = unit * (s.oh_top * ow + s.ow_left);
            assert(ic == input_c);

            for (int odi = 0; odi < output_d; ++odi) {
              int id_start = odi;
              int _kd = kd;
              int kernel_shift_idx = 0;

              // handle pd0
              if (odi == 0) {
                kd -= pad_d0;
                kernel_shift_idx = pad_d0;
              } else {
                id_start -= pad_d0;
              }

              // handle pd1
              if (odi == output_d - 1) {
                kd -= pad_d1;
              }

              for (int kdi = 0; kdi < kd; ++kdi) {
                int idi = id_start + kdi;
                int ps32_mode = get_ps32_mode(kdi, kd);

                cvk_tl_t tl_weight_tiu;
                initWeightForTiu(&tl_weight_tiu, tl_weight, oc_step, input_c,
                                 kh, kw, kdi + kernel_shift_idx, fmt);

                loadInput(layer_id, input_n, input_c, input_d, input_h, input_w,
                          idi, ga_ifmap + ifmap_offset, tl_input);
                compute(input_n, input_c, kh, kw, s.ph_top, s.ph_bot, s.pw_left,
                        s.pw_right, oc_pos, cur_oc, output_c, output_h,
                        output_w, stride_h, stride_w, ps32_mode, do_relu,
                        tl_input, &tl_weight_tiu, tl_bias, tl_output);
              }

              kd = _kd;

              storeOutput(oc_pos, cur_oc, output_c, output_d, output_h,
                          output_w, odi, ga_ofmap + ofmap_offset, tl_output);
            }
          }
        }
      }
    }
  }

  if (tl_bias)
    CV18xx::lmem_free_tensor(tl_bias);
  CV18xx::lmem_free_tensor(tl_weight);
  CV18xx::lmem_free_tensor(tl_input);
  CV18xx::lmem_free_tensor(tl_output);
}

} // namespace backend
} // namespace tpu_mlir

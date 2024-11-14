//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#define DEBUG_TYPE "match_template_kernel"

#define ASSERT(x) assert(x)

namespace tpu_mlir {
namespace backend {
static cvk_tl_t *load(cvk_tl_shape_t &tl_shape, cvk_tg_stride_t &gstride,
                      uint64_t ga_src, cvk_fmt_t ifmt, cvk_fmt_t fmt,
                      uint32_t layer_id, float_t mean) {
  cvk_tl_t *tl_ifmap = CV18xx::lmem_alloc_tensor(tl_shape, fmt, 1);
  ASSERT(tl_ifmap);
  cvk_tdma_g2l_tensor_copy_param_t p = {0};
  cvk_tg_t tg_src;
  tg_src.start_address = ga_src;
  tg_src.base_reg_index =
      CV18xx::getTdmaBaseSelectIndexFromGaddr(tg_src.start_address);
  tg_src.int8_rnd_mode = 0;
  tg_src.fmt = ifmt;
  tg_src.shape =
      CV18xx::tg_shape_t4(tl_shape.n, tl_shape.c, tl_shape.h, tl_shape.w);
  tg_src.stride = gstride;
  p.src = &tg_src;
  p.dst = tl_ifmap;
  CV18xx::tdma_g2l_tensor_copy(&p);
  // temporary fix overflow issue, for some extream case it still don't work
  if (mean != 0) {
    cvk_tiu_add_param_t p1 = {0};
    p1.res_high = nullptr;
    p1.res_low = tl_ifmap;
    p1.a_high = nullptr;
    p1.a_low = tl_ifmap;
    p1.b_is_const = true;
    p1.b_const.val = CV18xx::convert_fp32_to_bf16(-mean);
    p1.b_const.is_signed = 1;
    p1.rshift_bits = 0;
    p1.layer_id = layer_id;
    p1.relu_enable = false;
    CV18xx::tiu_add(&p1);
  }

  return tl_ifmap;
}

static cvk_tl_t *load_template(int64_t ga_src, int32_t c_step, int32_t h,
                               int32_t w, cvk_fmt_t ifmt, cvk_fmt_t fmt,
                               bool boradcast, uint32_t layer_id,
                               float_t mean) {
  cvk_tl_shape_t tl_tshape;
  if (boradcast)
    // load and broadcast template to lanes
    tl_tshape = CV18xx::tl_shape_t4(1, CV18xx::CV18xx::NPU_NUM, h, w);
  else
    tl_tshape = CV18xx::tl_shape_t4(1, c_step, h, w);
  cvk_tg_stride_t tg_tstride =
      CV18xx::tg_default_stride({1, 1, tl_tshape.h, tl_tshape.w}, ifmt);
  tg_tstride.n = 0;
  tg_tstride.c = 0;
  cvk_tl_t *tl_template =
      load(tl_tshape, tg_tstride, ga_src, ifmt, fmt, layer_id, mean);
  return tl_template;
}

// _____________         _____________
// ||slide |    |        |            |      output
// ||window|th  |        |     _______|     _____
// ||__tw__|   ih ...... |     |      | --> |    | n
// |            |        |     |      |     |____|
// |______iw____|        |_____|______|       c

void cvi_backend_tg_bf16_match_template_kernel(
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_template, gaddr_t ga_table,
    gaddr_t ga_mantissa_table, gaddr_t ga_output, int ih, int iw, int th,
    int tw, const char *mode) {
  int32_t n, c, h, w, stride, outer_size, reduce_size;
  uint32_t lmem_used = 0;
  cvk_fmt_t fmt = CVK_FMT_BF16;
  cvk_fmt_t ifmt = CVK_FMT_U8;
  int32_t g_elt_size = CV18xx::bytesize_of_fmt(ifmt);
  bool boradcast = false;

  // reshape input
  n = ih - th + 1;
  c = iw - tw + 1;
  h = th;
  w = tw;
  stride = iw;
  outer_size = n * c;
  reduce_size = h * w;

  if (h >= MAX_WIDTH || w >= MAX_WIDTH) {
    llvm::errs() << llvm::format("Template size[%d] is too large\n",
                                 reduce_size);
    assert(0);
  }
  // load table
  cvk_tl_shape_t table_shape = CV18xx::lut_table_shape(fmt);
  cvk_tl_t *tl_lut = CV18xx::lmem_alloc_tensor(table_shape, fmt, 1);
  cvk_tl_t *tl_lut_mantissa = CV18xx::lmem_alloc_tensor(table_shape, fmt, 1);
  CV18xx::tdma_load_table(tl_lut, ga_table);
  CV18xx::tdma_load_table(tl_lut_mantissa, ga_mantissa_table);
  lmem_used += 2 * CV18xx::lmem_tensor_to_size(table_shape, fmt, 1);
  // tile policy
  int c_step = std::min(outer_size, MAX_CHANNEL);
  while (c_step > 0) {
    // for table
    uint32_t mem_need = lmem_used;
    // for input
    mem_need += CV18xx::lmem_tensor_to_size(1, c_step, h, w, fmt, 1);
    // for intermidate value and output
    uint32_t out_mem_need =
        CV18xx::lmem_tensor_to_size(1, c_step, 1, 1, fmt, 1);
    if (!strcmp(mode, "TM_CCOEFF_NORMED")) {
      // for template. boradcast defult is false
      mem_need += CV18xx::lmem_tensor_to_size(1, c_step, h, w, fmt, 1);
      // 4 means tl_output, tl_buf, tl_lut_out, tl_lut_tpl_out
      mem_need += 4 * out_mem_need;
    } else if (!strcmp(mode, "TM_SQDIFF")) {
      // broadcast template,
      mem_need += CV18xx::lmem_tensor_to_size(1, CV18xx::NPU_NUM, h, w, fmt, 1);
      // for output, tl_buf, tl_lut_out
      mem_need += 3 * out_mem_need;
      boradcast = true;
    } else {
      llvm::errs() << llvm::format("Match template not support [%s] method.\n",
                                   mode);
      assert(0);
    }
    if (mem_need <= (uint32_t)CV18xx::LMEM_BYTES) {
      break;
    }
    if (c_step % CV18xx::NPU_NUM != 0) {
      c_step -= c_step % CV18xx::NPU_NUM;
    } else {
      c_step -= CV18xx::NPU_NUM;
    }
  }
  // Todo  parallel
  c_step = std::min(c_step, c);
  if (c_step <= 0) {
    llvm::errs() << llvm::format(
        "Tilling Match Template failed, src shape:[1,%d,%d,%d]\n", outer_size,
        h, w);
    assert(0);
  }

  cvk_tg_stride_t tg_istride = {
      (uint32_t)(stride * g_elt_size), (uint32_t)g_elt_size,
      (uint32_t)(stride * g_elt_size), (uint32_t)g_elt_size};

  cvk_tl_t *tl_template = load_template(ga_template, c_step, h, w, ifmt, fmt,
                                        boradcast, layer_id, 0);
  if (!strcmp(mode, "TM_CCOEFF_NORMED")) {
    cvk_tl_t *tl_tpl_mean =
        CV18xx::lmem_alloc_tensor(CV18xx::tl_shape_t4(1, c_step, 1, 1), fmt, 1);
    cvk_tiu_average_pooling_param_t m0 = {0};
    m0.ofmap = tl_tpl_mean;
    m0.ifmap = tl_template;
    m0.kh = h;
    m0.kw = w;
    m0.ins_h = 0;
    m0.ins_last_h = 0;
    m0.ins_w = 0;
    m0.ins_last_w = 0;
    m0.stride_h = 1;
    m0.stride_w = 1;
    m0.avg_pooling_const = CV18xx::convert_fp32_to_bf16(1.0);
    m0.ins_val = 0;
    m0.ins_fp = CV18xx::convert_fp32_to_bf16(0.0);
    m0.layer_id = layer_id;
    CV18xx::tiu_average_pooling(&m0);
    // cal template - mean
    tl_tpl_mean->shape = tl_template->shape;
    tl_tpl_mean->stride.h = 0;
    tl_tpl_mean->stride.w = 0;
    cvk_tiu_sub_param_t m1 = {0};
    m1.res_high = 0;
    m1.res_low = tl_template;
    m1.a_high = 0;
    m1.a_low = tl_template;
    m1.b_high = 0;
    m1.b_low = tl_tpl_mean;
    m1.rshift_bits = 0;
    m1.layer_id = layer_id;
    CV18xx::tiu_sub(&m1);
    CV18xx::lmem_free_tensor(tl_tpl_mean);

    // cal reduce_sum(pow(template, 2))
    cvk_tl_t *tl_output = CV18xx::lmem_alloc_tensor(
        CV18xx::tl_shape_t4(1, tl_template->shape.c, 1, 1), fmt, 1);
    cvk_tl_t *tl_buf =
        CV18xx::lmem_alloc_tensor(tl_output->shape, tl_output->fmt, 1);
    cvk_tl_t *tl_lut_tpl_out =
        CV18xx::lmem_alloc_tensor(tl_output->shape, tl_output->fmt, 1);
    cvk_tiu_depthwise_pt_convolution_param_t pt0 = {0};
    pt0.ofmap = tl_output;
    pt0.ifmap = tl_template;
    pt0.weight = tl_template;
    pt0.bias = nullptr;
    pt0.ins_h = 0;
    pt0.ins_w = 0;
    pt0.ins_last_h = 0;
    pt0.ins_last_w = 0;
    pt0.pad_bottom = 0;
    pt0.pad_top = 0;
    pt0.pad_left = 0;
    pt0.pad_right = 0;
    pt0.stride_h = 1;
    pt0.stride_w = 1;
    pt0.dilation_h = 1;
    pt0.dilation_w = 1;
    pt0.rshift_bits = 0;
    pt0.relu_enable = 0;
    pt0.layer_id = layer_id;
    CV18xx::tiu_pt_depthwise_convolution(&pt0);

    // lut reduce_sum(1 / sqrt(power(template, 2))) denominator0
    cvk_tiu_bf16_lookup_interp_table_param_t pt1 = {0};
    pt1.ifmap = tl_output;
    pt1.buf = tl_buf;
    pt1.tbl_answer = tl_lut;
    pt1.tbl_answer_mantissa = tl_lut_mantissa;
    pt1.ofmap = tl_lut_tpl_out;
    pt1.is_scientific = 1;
    CV18xx::tiu_bf16_lookup_interp_table(&pt1);

    CV18xx::parallel_disable();
    for (int c_pos = 0; c_pos < outer_size;) {
      int _c = std::min(c_step, outer_size - c_pos);
      uint64_t in_offset = (c_pos / c * stride + c_pos % c) * g_elt_size;
      uint64_t out_offset = c_pos * CV18xx::bytesize_of_fmt(fmt);
      if (c_pos / c != (c_pos + _c - 1) / c) {
        // if end idx in next row , cut off rest and next loop will cal from
        // next row.
        _c -= (c_pos + _c) % c; // num of windows keep
        c_pos += _c;
      } else {
        c_pos += c_step;
      }
      tl_output->shape.c = _c;
      tl_buf->shape.c = _c;
      tl_lut_tpl_out->shape.c = _c;
      tl_template->shape.c = _c;
      cvk_tl_t *tl_lut_out =
          CV18xx::lmem_alloc_tensor(tl_output->shape, tl_output->fmt, 1);
      // load input. For now input assume always be uint8
      cvk_tl_shape_t tl_ishape = CV18xx::tl_shape_t4(1, _c, h, w);
      cvk_tl_t *tl_input = load(tl_ishape, tg_istride, ga_input + in_offset,
                                ifmt, fmt, layer_id, 0);

      // cal input mean
      cvk_tiu_average_pooling_param_t p1 = {0};
      p1.ofmap = tl_buf;
      p1.ifmap = tl_input;
      p1.kh = h;
      p1.kw = w;
      p1.ins_h = 0;
      p1.ins_last_h = 0;
      p1.ins_w = 0;
      p1.ins_last_w = 0;
      p1.stride_h = 1;
      p1.stride_w = 1;
      p1.avg_pooling_const = CV18xx::convert_fp32_to_bf16(1.0);
      p1.ins_val = 0;
      p1.ins_fp = CV18xx::convert_fp32_to_bf16(0.0);
      p1.layer_id = layer_id;
      CV18xx::tiu_average_pooling(&p1);
      // cal input - mean
      tl_buf->shape = tl_template->shape;
      tl_buf->stride.h = 0;
      tl_buf->stride.w = 0;
      cvk_tiu_sub_param_t p2 = {0};
      p2.res_high = 0;
      p2.res_low = tl_input;
      p2.a_high = 0;
      p2.a_low = tl_input;
      p2.b_high = 0;
      p2.b_low = tl_buf;
      p2.rshift_bits = 0;
      p2.layer_id = layer_id;
      CV18xx::tiu_sub(&p2);
      // cal reduce_sum(pow(input - mean, 2))
      cvk_tiu_depthwise_pt_convolution_param_t p3 = {0};
      p3.ofmap = tl_output;
      p3.ifmap = tl_input;
      p3.weight = tl_input;
      p3.bias = nullptr;
      p3.ins_h = 0;
      p3.ins_w = 0;
      p3.ins_last_h = 0;
      p3.ins_last_w = 0;
      p3.pad_bottom = 0;
      p3.pad_top = 0;
      p3.pad_left = 0;
      p3.pad_right = 0;
      p3.stride_h = 1;
      p3.stride_w = 1;
      p3.dilation_h = 1;
      p3.dilation_w = 1;
      p3.rshift_bits = 0;
      p3.relu_enable = 0;
      p3.layer_id = layer_id;
      CV18xx::tiu_pt_depthwise_convolution(&p3);
      // cal numerator
      tl_buf->shape = tl_output->shape;
      tl_buf->stride = tl_output->stride;
      cvk_tiu_bf16_lookup_interp_table_param_t p4 = {0};
      p4.ifmap = tl_output;
      p4.buf = tl_buf;
      p4.tbl_answer = tl_lut;
      p4.tbl_answer_mantissa = tl_lut_mantissa;
      p4.ofmap = tl_lut_out;
      p4.is_scientific = 1;
      CV18xx::tiu_bf16_lookup_interp_table(&p4);
      // cal reduce_sum((inp-mean) * (tepl-mean))
      cvk_tiu_depthwise_pt_convolution_param_t p5 = {0};
      p5.ofmap = tl_output;
      p5.ifmap = tl_input;
      p5.weight = tl_template;
      p5.bias = nullptr;
      p5.ins_h = 0;
      p5.ins_w = 0;
      p5.ins_last_h = 0;
      p5.ins_last_w = 0;
      p5.pad_bottom = 0;
      p5.pad_top = 0;
      p5.pad_left = 0;
      p5.pad_right = 0;
      p5.stride_h = 1;
      p5.stride_w = 1;
      p5.dilation_h = 1;
      p5.dilation_w = 1;
      p5.rshift_bits = 0;
      p5.relu_enable = 0;
      p5.layer_id = layer_id;
      CV18xx::tiu_pt_depthwise_convolution(&p5);
      // mul numerator and denominator
      cvk_tiu_mul_param_t p6 = {0};
      p6.res_high = nullptr;
      p6.res_low = tl_output;
      p6.a = tl_output;
      p6.b = tl_lut_out;
      p6.b_is_const = 0;
      p6.rshift_bits = 0;
      p6.layer_id = layer_id;
      p6.relu_enable = 0;
      CV18xx::tiu_mul(&p6);

      cvk_tiu_mul_param_t p7 = {0};
      p7.res_high = nullptr;
      p7.res_low = tl_output;
      p7.a = tl_output;
      p7.b = tl_lut_tpl_out;
      p7.b_is_const = 0;
      p7.rshift_bits = 0;
      p7.layer_id = layer_id;
      p7.relu_enable = 0;
      CV18xx::tiu_mul(&p7);

      CV18xx::tdma_store(tl_output, ga_output + out_offset);
      CV18xx::lmem_free_tensor(tl_input);
      CV18xx::lmem_free_tensor(tl_lut_out);
    } // end for
    CV18xx::lmem_free_tensor(tl_lut_tpl_out);
    CV18xx::lmem_free_tensor(tl_buf);
    CV18xx::lmem_free_tensor(tl_output);
  } // end if
  else {
    ASSERT(0 && "Not support now.");
  } // end else
  // free table and template
  CV18xx::lmem_free_tensor(tl_template);
  CV18xx::lmem_free_tensor(tl_lut_mantissa);
  CV18xx::lmem_free_tensor(tl_lut);
} // end fun
} // namespace backend
} // namespace tpu_mlir

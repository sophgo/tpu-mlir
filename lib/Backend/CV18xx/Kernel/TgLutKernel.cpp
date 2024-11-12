//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include <llvm/Support/Debug.h>
#define DEBUG_TYPE "lut_kernel"

#define METHOD_MANTISSA 0
#define METHOD_LOG 1
#define METHOD_SLOPE 2

namespace tpu_mlir {
namespace backend {
void cvi_backend_tg_lut_kernel(uint32_t layer_id, gaddr_t ga_input,
                               gaddr_t ga_output, gaddr_t sg_lut_gaddr, int n,
                               int c, int h, int w, cvk_fmt_t fmt) {

  CV18xx::set_layer_id(layer_id);

  cvk_tl_shape_t table_shape = CV18xx::lut_table_shape(fmt);
  cvk_tl_t *sg_lut_table = CV18xx::lmem_alloc_tensor(table_shape, fmt, 1);
  CV18xx::tdma_load_table(sg_lut_table, sg_lut_gaddr);

  int blob_num = 1;
  uint32_t lmem_used = CV18xx::lmem_tensor_to_size(table_shape, fmt, 1);
  std::vector<CV18xx::tiling_info_t> tiles;
  CV18xx::tiling_packing(tiles, n, c, h, w, fmt, blob_num, lmem_used,
                         CV18xx::TilingAll);

  for (auto &tile : tiles) {
    cvk_tl_shape_t input_shape =
        CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    cvk_tl_t *bottom = CV18xx::lmem_alloc_tensor(input_shape, fmt, 1);
    CV18xx::tdma_load(bottom, ga_input + tile.offset);

    cvk_tiu_lookup_table_param_t p12 = {0};
    p12.ifmap = bottom;
    p12.ofmap = bottom; // it chould overwrite itself
    p12.table = sg_lut_table;
    p12.layer_id = layer_id;
    CV18xx::tiu_lookup_table(&p12);

    // move result to global
    CV18xx::tdma_store(bottom, ga_output + tile.offset);

    // free
    CV18xx::lmem_free_tensor(bottom);
  }
  CV18xx::lmem_free_tensor(sg_lut_table);
}

static void bf16_lut_slope_kernel(uint32_t layer_id, gaddr_t ga_input,
                                  gaddr_t ga_output, cvk_tl_t *tl_table_answer,
                                  cvk_tl_t *tl_table_answer_slope, int n, int c,
                                  int h, int w, gaddr_t gaddr_offset,
                                  float scale, cvk_fmt_t fmt) {

  cvk_tl_shape_t tl_shape = CV18xx::tl_shape_t4(n, c, h, w);

  // tl_ifmap reuse to input / output
  cvk_tl_t *tl_ifmap = CV18xx::lmem_alloc_tensor(tl_shape, fmt, 1);
  cvk_tl_t *tl_ofmap_slope = CV18xx::lmem_alloc_tensor(tl_shape, fmt, 1);
  cvk_tl_t *tl_ofmap_y0 = CV18xx::lmem_alloc_tensor(tl_shape, fmt, 1);

  CV18xx::tdma_load(tl_ifmap, ga_input + gaddr_offset);

  cvk_tdma_l2l_tensor_copy_param_t p3 = {0};
  // scale input for remap its idx(-x~x) to (-127~127), dirty tl_ifmap
  cvk_tiu_mul_param_t p4 = {0};
  p4.res_high = NULL;
  p4.res_low = tl_ifmap;
  p4.a = tl_ifmap;
  p4.b_is_const = 1;
  p4.b_const.val = CV18xx::convert_fp32_to_bf16(scale);
  p4.rshift_bits = 0;
  p4.relu_enable = 0;
  p4.layer_id = layer_id;
  CV18xx::tiu_mul(&p4);

  // <! get idx from bf16->int8
  memset(&p3, 0x00, sizeof(cvk_tdma_l2l_tensor_copy_param_t));
  cvk_tl_t dst = {};
  memcpy(&dst, tl_ofmap_y0, sizeof(cvk_tl_t));

  // we keep contiguous layout that we convert
  // it without 'gap' and leverage tiu_mv to
  // contiguous again
  // bf16->int8
  dst.shape = tl_ifmap->shape;
  dst.fmt = CVK_FMT_I8;
  dst.stride = CV18xx::tl_default_stride(dst.shape, dst.fmt, 1);
  dst.int8_rnd_mode = 1;
  p3.src = tl_ifmap;
  p3.dst = &dst;
  CV18xx::tdma_l2l_tensor_copy(&p3);
  dst.int8_rnd_mode = 0;

  // int8 to bf16
  p3.src = &dst;
  p3.dst = tl_ofmap_slope; //<! bf16
  CV18xx::tdma_l2l_tensor_copy(&p3);

  // <! sub, diff base , a - b
  // (x - x0)
  cvk_tiu_sub_param_t p5 = {0};
  p5.res_high = 0;
  p5.res_low = tl_ifmap;
  p5.a_high = 0;
  p5.a_low = tl_ifmap;
  p5.b_high = 0;
  p5.b_low = tl_ofmap_slope;
  p5.rshift_bits = 0;
  p5.layer_id = layer_id;
  CV18xx::tiu_sub(&p5);

  // move index seperate as bf16 size
  // copy to bf16 size
  {
    cvk_tl_t working = *tl_ofmap_slope;
    working.fmt = CVK_FMT_I8;
    cvk_tiu_copy_param_t param = {0};
    param.src = &dst;
    param.dst = &working;
    param.layer_id = layer_id;
    CV18xx::tiu_copy(&param);
    // back for next index

    dst.fmt = fmt;
    dst.shape = tl_ofmap_slope->shape;
    dst.stride = tl_ofmap_slope->stride;
    param.src = &working;
    param.dst = &dst;
    param.layer_id = layer_id;
    CV18xx::tiu_copy(&param);
  }
  // get f(x0) and slope(x)
  // reshape, 16->16
  dst.fmt = fmt;
  dst.shape = tl_ofmap_slope->shape;
  dst.stride = tl_ofmap_slope->stride;

  // <! get slope by index
  cvk_tiu_lookup_table_param_t p6 = {0};
  memset(&p6, 0x0, sizeof(cvk_tiu_lookup_table_param_t));
  p6.ofmap = tl_ofmap_slope;
  p6.ifmap = &dst;
  p6.table = tl_table_answer_slope;
  p6.layer_id = layer_id;
  CV18xx::tiu_lookup_table(&p6);

  // base f(x0)
  memset(&p6, 0x0, sizeof(cvk_tiu_lookup_table_param_t));
  p6.ofmap = tl_ofmap_y0;
  p6.ifmap = &dst;
  p6.table = tl_table_answer;
  p6.layer_id = layer_id;
  CV18xx::tiu_lookup_table(&p6);

  // <! mac
  // <! part A + part B, a * b + res = res
  cvk_tiu_mac_param_t p7 = {0};
  p7.res_high = 0;
  p7.res_low = tl_ofmap_y0;
  p7.res_is_int8 = 0;
  p7.a = tl_ifmap;
  p7.b_is_const = 0;
  p7.b = tl_ofmap_slope;
  p7.lshift_bits = 0; // lshift_bits;
  p7.rshift_bits = 0; // rshift_bits;
  p7.relu_enable = 0;
  p7.layer_id = layer_id;
  CV18xx::tiu_mac(&p7);

  CV18xx::tdma_store(tl_ofmap_y0, ga_output + gaddr_offset);
  CV18xx::lmem_free_tensor(tl_ofmap_y0);
  CV18xx::lmem_free_tensor(tl_ofmap_slope);
  CV18xx::lmem_free_tensor(tl_ifmap);
}

void cvi_backend_tg_bf16_lut_slope_kernel(uint32_t layer_id, gaddr_t ga_input,
                                          gaddr_t ga_output,
                                          gaddr_t y0_table_gaddr,
                                          gaddr_t slope_gaddr, int n, int c,
                                          int h, int w, float range_min,
                                          float range_max) {

  float scale = 256.0 / (range_max - range_min);

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "activation_kernel : ga_input %x ga_output %x "
                 "y0_table_gaddr %x slope_gaddr %x "
                 "n %d c %d h %d w %d scale %f\n",
                 ga_input, ga_output, y0_table_gaddr, slope_gaddr, n, c, h, w,
                 scale));

  cvk_tl_shape_t table_shape = CV18xx::lut_table_shape(CVK_FMT_BF16);
  cvk_tl_t *tl_table_answer =
      CV18xx::lmem_alloc_tensor(table_shape, CVK_FMT_BF16, 1);
  cvk_tl_t *tl_table_answer_slope =
      CV18xx::lmem_alloc_tensor(table_shape, CVK_FMT_BF16, 1);

  CV18xx::tdma_load_table(tl_table_answer, y0_table_gaddr);
  CV18xx::tdma_load_table(tl_table_answer_slope, slope_gaddr);

  int blob_num = 3;
  uint32_t lmem_used =
      2 * CV18xx::lmem_tensor_to_size(table_shape, CVK_FMT_BF16, 1);
  std::vector<CV18xx::tiling_info_t> tiles;
  CV18xx::tiling_packing(tiles, n, c, h, w, CVK_FMT_BF16, blob_num, lmem_used,
                         CV18xx::TilingAll);

  for (auto &tile : tiles) {
    bf16_lut_slope_kernel(layer_id, ga_input, ga_output, tl_table_answer,
                          tl_table_answer_slope, tile.n, tile.c, tile.h, tile.w,
                          tile.offset, scale, CVK_FMT_BF16);
  }

  CV18xx::lmem_free_tensor(tl_table_answer_slope);
  CV18xx::lmem_free_tensor(tl_table_answer);
}

void cvi_backend_tg_bf16_lut_mantissa_kernel(uint32_t layer_id,
                                             gaddr_t ga_input,
                                             gaddr_t ga_output,
                                             gaddr_t exp_lut_table,
                                             gaddr_t mantissa_lut_table, int n,
                                             int c, int h, int w, int method) {

  cvk_fmt_t fmt = CVK_FMT_BF16;
  cvk_tl_shape_t table_shape = CV18xx::lut_table_shape(CVK_FMT_BF16);

  cvk_tl_t *tl_table_answer = CV18xx::lmem_alloc_tensor(table_shape, fmt, 1);
  cvk_tl_t *tl_table_answer_mantissa =
      CV18xx::lmem_alloc_tensor(table_shape, fmt, 1);

  // load exp / mantissa table
  CV18xx::tdma_load_table(tl_table_answer, exp_lut_table);
  CV18xx::tdma_load_table(tl_table_answer_mantissa, mantissa_lut_table);

  int blob_num = 3;
  uint32_t lmem_used =
      2 * CV18xx::lmem_tensor_to_size(table_shape, CVK_FMT_BF16, 1);
  std::vector<CV18xx::tiling_info_t> tiles;
  CV18xx::tiling_packing(tiles, n, c, h, w, CVK_FMT_BF16, blob_num, lmem_used,
                         CV18xx::TilingAll);
  CV18xx::parallel_disable();
  for (auto &tile : tiles) {

    cvk_tl_shape_t slice_shape =
        CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);

    // alloc local memory
    cvk_tl_t *tl_ifmap = CV18xx::lmem_alloc_tensor(slice_shape, fmt, 1);
    cvk_tl_t *tl_buf = CV18xx::lmem_alloc_tensor(slice_shape, fmt, 1);
    cvk_tl_t *tl_ofmap = CV18xx::lmem_alloc_tensor(slice_shape, fmt, 1);

    // load input
    CV18xx::tdma_load(tl_ifmap, ga_input + tile.offset);
    if (method == METHOD_MANTISSA) {
      cvk_tiu_bf16_lookup_interp_table_param_t param = {0};
      param.ifmap = tl_ifmap;
      param.buf = tl_buf;
      param.tbl_answer = tl_table_answer;
      param.tbl_answer_mantissa = tl_table_answer_mantissa;
      param.ofmap = tl_ofmap;
      param.is_scientific = 1;
      CV18xx::tiu_bf16_lookup_interp_table(&param);
    } else if (method == METHOD_LOG) {
      cvi_backend_bf16_tl_log_lut_mantissa_method(
          layer_id, tl_ifmap->start_address, tl_ofmap->start_address,
          tl_buf->start_address, tl_table_answer->start_address,
          tl_table_answer_mantissa->start_address, tile.n, tile.c, tile.h,
          tile.w, false);
    }
    CV18xx::tdma_store(tl_ofmap, ga_output + tile.offset);

    CV18xx::lmem_free_tensor(tl_ofmap);
    CV18xx::lmem_free_tensor(tl_buf);
    CV18xx::lmem_free_tensor(tl_ifmap);
  }

  CV18xx::lmem_free_tensor(tl_table_answer_mantissa);
  CV18xx::lmem_free_tensor(tl_table_answer);
}

} // namespace backend
} // namespace tpu_mlir

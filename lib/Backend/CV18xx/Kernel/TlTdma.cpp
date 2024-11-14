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

#define DEBUG_TYPE "tl_tdma"

namespace tpu_mlir {
namespace backend {
void cvi_backend_tl_to_tensor(

    cvk_tl_t *tensor, laddr_t la, uint32_t tensor_n, uint32_t tensor_c,
    uint32_t tensor_h, uint32_t tensor_w, cvk_fmt_t fmt, uint8_t eu_align) {
  CV18xx::assert_support_fmt(fmt);
  tensor->start_address = la;
  tensor->fmt = fmt;
  tensor->shape = CV18xx::tl_shape_t4(tensor_n, tensor_c, tensor_h, tensor_w);
  tensor->stride = CV18xx::tl_default_stride(tensor->shape, fmt, eu_align);
}

void cvi_backend_tl_load_stride(uint32_t layer_id, gaddr_t ga_src,
                                laddr_t la_dst, int Local_N, int Local_C,
                                int Local_H, int Local_W, int Global_C,
                                int Global_H, int Global_W, bool DoTranspose,
                                bool DoAligned, bool isNeuron, cvk_fmt_t from,
                                cvk_fmt_t to) {

  cvi_backend_tl_load_stride(layer_id, ga_src, la_dst, Local_N, Local_C,
                             Local_H, Local_W, Global_C, Global_H, Global_W,
                             DoTranspose, DoAligned, isNeuron, from, to,
                             false // DoDecompress
  );
}

void cvi_backend_tl_load_compressed(uint32_t layer_id, gaddr_t ga_src,
                                    laddr_t la_dst, int Local_N, int Local_C,
                                    int Local_H, int Local_W, int Global_C,
                                    int Global_H, int Global_W,
                                    bool DoTranspose, bool DoAligned,
                                    bool isNeuron, cvk_fmt_t from, cvk_fmt_t to,
                                    int h_step, int step_size, int c_step) {

  // Global shape is used for stride - global memory layout

  int eu_align = DoAligned ? 1 : 0;

  cvk_tl_stride_t tl_stride = CV18xx::tl_default_stride(
      CV18xx::tl_shape_t4(Local_N, Local_C, Local_H, Local_W), to, eu_align);

  uint32_t tl_cmpr_block_c_stride =
      CV18xx::tl_cmpr_c_stride(1, c_step, Local_H, Local_W, to);

  for (int i = 0; i < Local_C; i += c_step) {
    int cur_c = std::min(c_step, Local_C - i);
    uint32_t la_cmpr_start =
        CV18xx::addr_after_right_shift(la_dst, i, tl_cmpr_block_c_stride);
    for (int j = 0; j < Local_H; j += h_step) {
      int cur_h = std::min(h_step, Local_H - j);

      // Output HxW is contigious in each lane, eu_align = 0
      cvk_tl_shape_t tl_tiled_shape =
          CV18xx::tl_shape_t4(1, cur_c, cur_h, Local_W);

      cvk_tl_t tl_tiled_dst;
      CV18xx::lmem_init_tensor(&tl_tiled_dst, tl_tiled_shape, to, eu_align);
      tl_tiled_dst.stride = tl_stride;
      tl_tiled_dst.start_address = la_cmpr_start + j * tl_stride.h;

      uint32_t ga_cmpr_offset =
          CV18xx::ga_cmpr_offset(Local_N, Global_C, Global_H, Global_W, 0, i,
                                 j / h_step, c_step, step_size);

      // (1, CV18xx::NPU_NUM, 1, w) ... (1, CV18xx::NPU_NUM, 1, w).
      cvk_tg_shape_t tg_tiled_shape =
          CV18xx::tg_shape_t4(1, cur_c, cur_h, Local_W);

      cvk_cmpr_tg_t tg_cmpr_src = {0};
      CV18xx::gmem_init_tensor(&tg_cmpr_src.t, tg_tiled_shape, from);
      tg_cmpr_src.t.base_reg_index =
          CV18xx::getTdmaBaseSelectIndexFromGaddr(ga_src);
      tg_cmpr_src.t.start_address = ga_src + ga_cmpr_offset;

      cvk_tdma_g2l_tensor_copy_decompressed_param_t param = {0};
      param.src = &tg_cmpr_src;
      param.dst = &tl_tiled_dst;
      param.layer_id = layer_id;
      CV18xx::tdma_g2l_tensor_copy_decompressed(&param);
    }
  }
}

void cvi_backend_tl_load_stride(uint32_t layer_id, gaddr_t ga_src,
                                laddr_t la_dst, int Local_N, int Local_C,
                                int Local_H, int Local_W, int Global_C,
                                int Global_H, int Global_W, bool DoTranspose,
                                bool DoAligned, bool isNeuron, cvk_fmt_t from,
                                cvk_fmt_t to, bool DoDecompress) {
  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "cvi_backend_tl_load_stride:\n"
                 "    layer_id %d\n"
                 "    src (, %d, %d, %d), dst (%d, %d, %d, %d)\n"
                 "    src 0x%lx, dst 0x%lx\n"
                 "    DoTranspose %d, DoAligned %d, isNeuron %d\n"
                 "    from %d to %d, DoDeCompress %d\n",
                 layer_id, Global_C, Global_H, Global_W, Local_N, Local_C,
                 Local_H, Local_W, ga_src, la_dst, DoTranspose, DoAligned,
                 isNeuron, from, to, DoDecompress));
  // tensor in local memory
  cvk_tl_shape_t tl_shape;
  tl_shape.n = DoTranspose ? Local_C : Local_N;
  tl_shape.c = DoTranspose ? Local_N : Local_C;
  tl_shape.h = Local_H;
  tl_shape.w = Local_W;

  if (DoDecompress) {
    assert(((from == CVK_FMT_I8 && to == CVK_FMT_I8) ||
            (from == CVK_FMT_BF16 && to == CVK_FMT_BF16)) &&
           "Only support i8/bf16 now");
  }

  cvk_tl_t tl_data;
  tl_data.start_address = la_dst;
  tl_data.fmt = from;
  tl_data.shape = tl_shape;
  tl_data.stride =
      CV18xx::tl_default_stride(tl_shape, tl_data.fmt, DoAligned ? 1 : 0);

  // Global shape used for stride calculation
  cvk_tg_stride_t ga_stride =
      CV18xx::tg_default_stride({(uint32_t)Local_N, (uint32_t)Global_C,
                                 (uint32_t)Global_H, (uint32_t)Global_W},
                                from);
  CV18xx::tdma_load_stride(&tl_data, ga_src, ga_stride, false, DoDecompress);
}

void cvi_backend_tl_load_stride_broadcast(uint32_t layer_id, gaddr_t ga_src,
                                          laddr_t la_dst, int Local_N,
                                          int Local_C, int Local_H, int Local_W,
                                          int Global_C, int Global_H,
                                          int Global_W, bool DoAligned,
                                          bool isNeuron, cvk_fmt_t from,
                                          cvk_fmt_t to, bool DoDecompress) {
  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "cvi_backend_tl_load_stride:\n"
                 "    layer_id %d\n"
                 "    src (, %d, %d, %d), dst (%d, %d, %d, %d)\n"
                 "    src 0x%lx, dst 0x%lx\n"
                 "    DoAligned %d, isNeuron %d\n"
                 "    from %d to %d, DoDeCompress %d\n",
                 layer_id, Global_C, Global_H, Global_W, Local_N, Local_C,
                 Local_H, Local_W, ga_src, la_dst, DoAligned, isNeuron, from,
                 to, DoDecompress));
  Local_C = Local_C == 1 ? CV18xx::NPU_NUM : Local_C;
  Global_C = Global_C == 1 ? CV18xx::NPU_NUM : Global_C;
  // tensor in local memory
  cvk_tl_shape_t tl_shape;
  tl_shape.n = Local_N;
  tl_shape.c = Local_C;
  tl_shape.h = Local_H;
  tl_shape.w = Local_W;

  if (DoDecompress) {
    assert(((from == CVK_FMT_I8 && to == CVK_FMT_I8) ||
            (from == CVK_FMT_BF16 && to == CVK_FMT_BF16)) &&
           "Only support i8/bf16 now");
  }

  cvk_tl_t tl_data;
  tl_data.start_address = la_dst;
  tl_data.fmt = from;
  tl_data.shape = tl_shape;
  tl_data.stride =
      CV18xx::tl_default_stride(tl_shape, tl_data.fmt, DoAligned ? 1 : 0);

  // Global shape used for stride calculation
  cvk_tg_stride_t ga_stride =
      CV18xx::tg_default_stride({(uint32_t)Local_N, (uint32_t)Global_C,
                                 (uint32_t)Global_H, (uint32_t)Global_W},
                                from);
  ga_stride.c = 0;
  ga_stride.n = 0;
  CV18xx::tdma_load_stride(&tl_data, ga_src, ga_stride, false, DoDecompress);
}

void cvi_backend_tl_load(uint32_t layer_id, laddr_t la_ifmap, gaddr_t ga_ifmap,
                         cvk_fmt_t fmt, uint32_t n, uint32_t ic, uint32_t ih,
                         uint32_t iw, bool do_decompress) {
  cvk_tl_t tl_ifmap;
  tl_ifmap.start_address = la_ifmap;
  tl_ifmap.fmt = fmt;
  tl_ifmap.shape = CV18xx::tl_shape_t4(n, ic, ih, iw);
  tl_ifmap.stride =
      CV18xx::tl_default_stride(tl_ifmap.shape, fmt, /*eu_align=*/1);
  CV18xx::set_layer_id(layer_id);

  if (fmt == CVK_FMT_I8 || fmt == CVK_FMT_U8) {
    cvk_tg_stride_t ifmap_gstride = {ic * ih * iw, ih * iw, iw};
    CV18xx::tdma_load_stride(&tl_ifmap, ga_ifmap, ifmap_gstride,
                             /*do_transpose=*/false, do_decompress);
  } else if (fmt == CVK_FMT_BF16) {
    CV18xx::tdma_load(&tl_ifmap, ga_ifmap);
  } else {
    assert(0);
  }
}

void cvi_backend_tl_load(uint32_t layer_id, laddr_t la_ifmap, gaddr_t ga_ifmap,
                         cvk_fmt_t fmt, uint32_t n, uint32_t ic, uint32_t ih,
                         uint32_t iw) {
  cvi_backend_tl_load(layer_id, la_ifmap, ga_ifmap, fmt, n, ic, ih, iw,
                      /*do_decompress=*/false);
}

void cvi_backend_tl_store(uint32_t layer_id, laddr_t la_ofmap, gaddr_t ga_ofmap,
                          cvk_fmt_t fmt, uint32_t n, uint32_t oc, uint32_t oh,
                          uint32_t ow) {
  cvk_tl_t tl_ofmap;
  tl_ofmap.start_address = la_ofmap;
  tl_ofmap.fmt = fmt;
  tl_ofmap.shape = CV18xx::tl_shape_t4(n, oc, oh, ow);
  tl_ofmap.stride =
      CV18xx::tl_default_stride(tl_ofmap.shape, fmt, /*eu_align=*/1);

  CV18xx::set_layer_id(layer_id);
  CV18xx::tdma_store(&tl_ofmap, ga_ofmap);
}

void cvi_backend_tl_store_stride(uint32_t layer_id, gaddr_t ga_dst,
                                 laddr_t la_src, int Local_N, int Local_C,
                                 int Local_H, int Local_W, int Global_C,
                                 int Global_H, int Global_W, bool DoTranspose,
                                 bool DoAligned, bool isNeuron, cvk_fmt_t from,
                                 cvk_fmt_t to, bool DoCompress) {
  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "cvi_backend_tl_store_stride:\n"
                 "    layer_id %d\n"
                 "    src (%d, %d, %d, %d), dst (, %d, %d, %d)\n"
                 "    src 0x%lx, dst 0x%lx\n"
                 "    DoTranspose %d, DoAligned %d, isNeuron %d"
                 "    from %d to %d, DoCompress %d\n",
                 layer_id, Local_N, Local_C, Local_H, Local_W, Global_C,
                 Global_H, Global_W, la_src, ga_dst, DoTranspose, DoAligned,
                 isNeuron, from, to, DoCompress));
  // tensor in local memory
  cvk_tl_shape_t tl_shape;
  tl_shape.n = DoTranspose ? Local_C : Local_N;
  tl_shape.c = DoTranspose ? Local_N : Local_C;
  tl_shape.h = Local_H;
  tl_shape.w = Local_W;

  cvk_tl_t tl_data;
  tl_data.start_address = la_src;
  tl_data.fmt = from;
  tl_data.shape = tl_shape;
  tl_data.stride =
      CV18xx::tl_default_stride(tl_shape, tl_data.fmt, DoAligned ? 1 : 0);

  cvk_tg_stride_t ts_stride =
      CV18xx::tg_default_stride({(uint32_t)Local_N, (uint32_t)Global_C,
                                 (uint32_t)Global_H, (uint32_t)Global_W},
                                to);

  // We need another API to pass memory region from TPU dialect codegen.
  CV18xx::tdma_store_stride(&tl_data, ga_dst, ts_stride,
                            /*do_transpose=*/false, DoCompress);
}

void cvi_backend_tl_store_stride(uint32_t layer_id, gaddr_t ga_dst,
                                 laddr_t la_src, int Local_N, int Local_C,
                                 int Local_H, int Local_W, int Global_C,
                                 int Global_H, int Global_W, bool DoTranspose,
                                 bool DoAligned, bool isNeuron, cvk_fmt_t from,
                                 cvk_fmt_t to) {

  cvi_backend_tl_store_stride(
      layer_id, ga_dst, la_src, Local_N, Local_C, Local_H, Local_W, Global_C,
      Global_H, Global_W, DoTranspose, DoAligned, isNeuron, from, to, false);
}

// Tiled compressed activation split as (n, c_step, h_step, w)
// Global memory layout: (n, c/c_step, h/h_step, c_step, h_step, w)
//
// output shape       (1, 64, 35, 112)
// tiled output shape (1, 64,  1, 112)
//
// tiled TDMA store
//   (1, 64, 1, 112)
//   (1, 64, 1, 112)
//   ...
//   (1, 64, 1, 112)
//
//  n   h  c  w
//  0   0  0  0       | header | compressed shape (1, h=1, c=64, w=112) |
//  0   1  0  0
//
//  0  34  0  0
//
void cvi_backend_tl_store_compressed(
    uint32_t layer_id, gaddr_t ga_dst, laddr_t la_src, int Local_N, int Local_C,
    int Local_H, int Local_W, int Global_C, int Global_H, int Global_W,
    bool DoTranspose, bool DoAligned, bool isNeuron, cvk_fmt_t from,
    cvk_fmt_t to, int h_step, int step_size, int c_step, bool DoIntraCmdParal) {

  // Global shape is used for stride - global memory layout
  assert(from == to && "Expect same data type");

  int eu_align = DoAligned ? 1 : 0;
  cvk_tl_stride_t tl_stride = CV18xx::tl_default_stride(
      CV18xx::tl_shape_t4(Local_N, Local_C, Local_H, Local_W), from, eu_align);

  uint32_t tl_cmpr_block_c_stride =
      CV18xx::tl_cmpr_c_stride(1, c_step, Local_H, Local_W, from);
  for (int i = 0; i < Local_C; i += c_step) {
    int cur_c = std::min(c_step, Local_C - i);
    uint32_t la_cmpr_start =
        CV18xx::addr_after_right_shift(la_src, i, tl_cmpr_block_c_stride);
    for (int j = 0; j < Local_H; j += h_step) {
      int cur_h = std::min(h_step, Local_H - j);

      // Output HxW is contigious in each lane, eu_align = 0
      cvk_tl_shape_t tl_tiled_shape =
          CV18xx::tl_shape_t4(1, cur_c, cur_h, Local_W);
      cvk_tl_t tl_tiled_src;
      CV18xx::lmem_init_tensor(&tl_tiled_src, tl_tiled_shape, from, eu_align);
      tl_tiled_src.stride = tl_stride;
      tl_tiled_src.start_address = la_cmpr_start + j * tl_stride.h;

      uint32_t ga_cmpr_offset =
          CV18xx::ga_cmpr_offset(Local_N, Global_C, Global_H, Global_W, 0, i,
                                 j / h_step, c_step, step_size);

      // (1, CV18xx::NPU_NUM, 1, w) ... (1, CV18xx::NPU_NUM, 1, w).
      cvk_tg_shape_t tg_tiled_shape =
          CV18xx::tg_shape_t4(1, cur_c, cur_h, Local_W);
      cvk_cmpr_tg_t tg_cmpr_dst = {0};
      tg_cmpr_dst.bias0 = (to == CVK_FMT_BF16) ? 127 : 0;
      CV18xx::gmem_init_tensor(&tg_cmpr_dst.t, tg_tiled_shape, to);
      tg_cmpr_dst.t.base_reg_index =
          CV18xx::getTdmaBaseSelectIndexFromGaddr(ga_dst);
      tg_cmpr_dst.t.start_address = ga_dst + ga_cmpr_offset;

      cvk_tdma_l2g_tensor_copy_compressed_param_t param = {0};
      param.src = &tl_tiled_src;
      param.dst = &tg_cmpr_dst;
      param.intra_cmd_paral = DoIntraCmdParal ? 1 : 0;
      CV18xx::tdma_l2g_tensor_copy_compressed(&param);
    }
  }
}

void cvi_backend_tl_copy(uint32_t layer_id, int la_src, int la_dst, int n,
                         int c, int h, int w, bool align, cvk_fmt_t fmt) {

  LLVM_DEBUG(llvm::errs() << llvm::format("cvi_backend_tl_copy:\n"
                                          "    layer_id %d\n"
                                          "    shape (%d, %d, %d, %d)\n"
                                          "    src 0x%lx, dst 0x%lx\n"
                                          "    DoAligned %d, fmt: %d\n",
                                          layer_id, n, c, h, w, la_src, la_dst,
                                          align, fmt););

  CV18xx::set_layer_id(layer_id);

  cvk_tl_t tl_dst = {};
  cvk_tl_t tl_src = {};

  tl_src.start_address = la_src;
  tl_src.fmt = fmt;
  tl_src.shape = CV18xx::tl_shape_t4(n, c, h, w);
  tl_src.stride = CV18xx::tl_default_stride(tl_src.shape, fmt, align);

  tl_dst.start_address = la_dst;
  tl_dst.fmt = fmt;
  tl_dst.shape = CV18xx::tl_shape_t4(n, c, h, w);
  tl_dst.stride = CV18xx::tl_default_stride(tl_dst.shape, fmt, align);

  cvk_tdma_l2l_tensor_copy_param_t p0 = {0};
  p0.dst = &tl_dst;
  p0.src = &tl_src;
  CV18xx::tdma_l2l_tensor_copy(&p0);
}

void cvi_backend_tl_bf16_ps32_to_fp32(uint32_t layer_id, laddr_t la_addr, int n,
                                      int c, int h, int w) {
  assert((n > 1) && ((n % 2) == 0) && "Expect ps32 shape");
  assert((h == 1) && (w == 1) && "Only support h=1, w=1");
  n /= 2; // Exclude lower part

  int eu_align = 1; // the result of tiu operation always align eu
  cvk_fmt_t fmt = CVK_FMT_BF16;
  cvk_tl_shape_t shape = CV18xx::tl_shape_t4(n, c, h, w);
  cvk_tl_stride_t stride = CV18xx::tl_default_stride(shape, fmt, eu_align);

  uint32_t la_high = la_addr;
  uint32_t la_low = la_addr + stride.n * n;

  cvk_tl_t tl_src = {};
  CV18xx::lmem_init_tensor(&tl_src, shape, fmt, eu_align);
  tl_src.start_address = la_high;
  tl_src.shape = shape;
  tl_src.stride = {stride.n, (uint32_t)CV18xx::EU_BYTES, stride.h, stride.w};

  cvk_tl_t tl_dst = {};
  CV18xx::lmem_init_tensor(&tl_dst, shape, fmt, eu_align);
  tl_dst.start_address = la_low + sizeof(uint16_t); // concat higher part
  tl_dst.shape = shape;
  tl_dst.stride = {stride.n, (uint32_t)CV18xx::EU_BYTES, stride.h, stride.w};

  cvk_tdma_l2l_tensor_copy_param_t param = {0};
  param.src = &tl_src;
  param.dst = &tl_dst;
  param.layer_id = layer_id;
  CV18xx::tdma_l2l_tensor_copy(&param);
}

void cvi_backend_tl_store_fp32(uint32_t layer_id, gaddr_t ga_dst,
                               laddr_t la_src, int n, int c, int h, int w) {
  n /= 2; // Exclude lower part

  int eu_align = 1; // the result of tiu operation always align eu
  cvk_fmt_t fmt = CVK_FMT_BF16;
  cvk_tl_shape_t shape = CV18xx::tl_shape_t4(n, c, h, w);
  cvk_tl_stride_t stride = CV18xx::tl_default_stride(shape, fmt, eu_align);

  uint32_t la_low = la_src + stride.n * n;

  cvk_tl_t tl_src = {0};
  tl_src.start_address = la_low;
  tl_src.fmt = fmt;
  tl_src.shape =
      CV18xx::tl_shape_t4(n, c, h, (sizeof(uint32_t) / sizeof(uint16_t)) * w);
  tl_src.stride = {stride.n, (uint32_t)CV18xx::EU_BYTES, stride.h, stride.w};

  cvk_tg_shape_t tg_shape = {(uint32_t)n, (uint32_t)c, (uint32_t)h,
                             (uint32_t)(2 * w)};
  cvk_tg_t tg_dst = {0};
  CV18xx::gmem_init_tensor(&tg_dst, tg_shape, fmt);
  tg_dst.base_reg_index = CV18xx::getTdmaBaseSelectIndexFromGaddr(ga_dst);
  tg_dst.start_address = ga_dst;
  // tg_dst.stride.c = 2;

  cvk_tdma_l2g_tensor_copy_param_t param = {0};
  param.src = &tl_src;
  param.dst = &tg_dst;
  CV18xx::tdma_l2g_tensor_copy(&param);
}
} // namespace backend
} // namespace tpu_mlir

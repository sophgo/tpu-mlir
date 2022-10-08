//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//


/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 */


#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/raw_ostream.h>
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Support/Helper/Module.h"

#define DEBUG_TYPE "CviBackendContext"

using namespace tpu_mlir::helper;
namespace tpu_mlir {
namespace backend {
// constexpr llvm::StringRef CV18xx::LIB_NAME;
// use local
#undef NPU_NUM
#undef EU_NUM
#undef LOCAL_MEM_SIZE
#undef LOCAL_MEM_BANKS
#define NPU_NUM cvi_chip_info_context(CVI_CHIP_LANE_NUM)
#define EU_NUM cvi_chip_info_context(CVI_CHIP_EU_NUM)
#define LOCAL_MEM_SIZE cvi_chip_info_context(CVI_CHIP_LMEM_SIZE)
#define LOCAL_MEM_BANKS cvi_chip_info_context(CVI_CHIP_LMEM_BANK)

CviBackendContext::CviBackendContext(const char *runchip) {
  cvk_cmd_buf_.reserve(0x20000000);
  cvk_reg_info_t req_info;
  strncpy(req_info.chip_ver_str, runchip, sizeof(req_info.chip_ver_str) - 1);
  req_info.cmdbuf_size = cvk_cmd_buf_.capacity();
  req_info.cmdbuf = cvk_cmd_buf_.data();

  cvk_ctx_ = CV18xx::instance().dl_cvikernel_register(&req_info);

  // Default mapping between tdma base selection
  // and global memory region.
  tdmaBaseSelects[SHARED_MEMORY] = 0;
  tdmaBaseSelects[WEIGHT_MEMORY] = 1;
  tdmaBaseSelects[PRIVATE_MEMORY] = 2;
  tdmaBaseSelects[IO_MEMORY_0] = 3;
  tdmaBaseSelects[IO_MEMORY_1] = 4;
  tdmaBaseSelects[IO_MEMORY_2] = 5;
  tdmaBaseSelects[IO_MEMORY_3] = 6;
  tdmaBaseSelects[IO_MEMORY_4] = 7;

  LLVM_DEBUG(llvm::errs() << "register " << runchip << " done\n";);
}

CviBackendContext::~CviBackendContext() {
  cvk_ctx_->ops->cleanup(cvk_ctx_);
  cvk_cmd_buf_.clear();
  cvk_cmd_buf_.shrink_to_fit();
  free(cvk_ctx_);
}

void CviBackendContext::write_cmdbuf(const void *cmdbuf, uint32_t size) {
  cmdbuf_.resize(size);
  memcpy(&cmdbuf_[0], cmdbuf, size);
}

void CviBackendContext::read_cmdbuf(std::vector<uint8_t> &out_cmdbuf) {
  out_cmdbuf.assign(cmdbuf_.begin(), cmdbuf_.end());
}

void CviBackendContext::dmabuf_convert(std::vector<uint8_t> &dmabuf) {
  uint32_t dmabuf_sz = 0;
  uint32_t pmu_sz = 0;
  cvk_ctx_->ops->dmabuf_size(cmdbuf_.data(), cmdbuf_.size(),
                             &dmabuf_sz, &pmu_sz);
  dmabuf.resize(dmabuf_sz);
  cvk_ctx_->ops->dmabuf_convert(cmdbuf_.data(), cmdbuf_.size(),
                                dmabuf.data());
}

void CviBackendContext::submit() {
  uint32_t size;
  uint8_t *cmdbuf = cvk_ctx_->ops->acquire_cmdbuf(cvk_ctx_, &size);
  write_cmdbuf(cmdbuf, size);
  cvk_ctx_->ops->reset(cvk_ctx_);
}

int CviBackendContext::cvi_chip_info_context(
    CVI_CHIP_INFO_E cvi_chip_info_e) const {
  if (cvi_chip_info_e == CVI_CHIP_VERSION)
    return cvk_ctx_->info.version;
  else if (cvi_chip_info_e == CVI_CHIP_NODE_SHIFT)
    return cvk_ctx_->info.node_shift;
  else if (cvi_chip_info_e == CVI_CHIP_LANE_NUM)
    return cvk_ctx_->info.npu_num;
  else if (cvi_chip_info_e == CVI_CHIP_LANE_SHIFT)
    return cvk_ctx_->info.npu_shift;
  else if (cvi_chip_info_e == CVI_CHIP_EU_NUM)
    return cvk_ctx_->info.eu_num;
  else if (cvi_chip_info_e == CVI_CHIP_EU_SHIFT)
    return cvk_ctx_->info.eu_shift;
  else if (cvi_chip_info_e == CVI_CHIP_LMEM_SIZE)
    return cvk_ctx_->info.lmem_size;
  else if (cvi_chip_info_e == CVI_CHIP_LMEM_BANK)
    return cvk_ctx_->info.lmem_banks;
  else
    assert(0);
}

uint8_t
CviBackendContext::getTdmaBaseSelectIndexFromGaddr(gaddr_t gaddr) const {
  // we store memory region value in bits (40 ~ 42) of gaddr;
  uint32_t memoryRegion = ((((uint64_t)gaddr) >> 40) & 0x07);
  if (memoryRegion < MAX_GLOBAL_MEMORY_REGION) {
    return tdmaBaseSelects[memoryRegion];
  }
  return 0;
}

CviBackendContext *cvi_backend_create_context(const char *runchip) {
  CviBackendContext *ctx = new CviBackendContext(runchip);
  return ctx;
}

void cvi_backend_delete_context(CviBackendContext *ctx) {
  delete ctx;
}

void cvi_backend_submit(CviBackendContext *ctx) {
  ctx->submit();
}

void cvi_backend_get_cmdbuf(CviBackendContext *ctx,
                            std::vector<uint8_t> &cmdbuf) {
  ctx->read_cmdbuf(cmdbuf);
}

void cvi_backend_dmabuf_convert(CviBackendContext *ctx,
                                std::vector<uint8_t> &dmabuf) {
  ctx->dmabuf_convert(dmabuf);
}

void cvi_backend_parallel_enable(CviBackendContext *ctx) {
  ctx->parallel_enable();
}

void cvi_backend_parallel_disable(CviBackendContext *ctx) {
  ctx->parallel_disable();
}

int cvi_backend_chip_context(CviBackendContext *ctx,
                             CVI_CHIP_INFO_E cvi_chip_info_e) {
  return ctx->cvi_chip_info_context(cvi_chip_info_e);
}

void cvi_backend_set_layer_id(CviBackendContext *ctx, int layer_id) {
  ctx->set_layer_id(layer_id);
}

// tdma api

void CviBackendContext::tdma_load_stride(cvk_tl_t *tlp, uint64_t ga_src,
                                         cvk_tg_stride_t ts_stride,
                                         bool do_transpose,
                                         bool do_decompress) const {
  assert(tlp != nullptr);

  // tensor in system memory
  //
  // Constraint:
  //   assert_tl_tg_same_size()
  //   Global_N == Local_N
  //
  // 1. Global channel != local channel
  //    Eg.
  //     alexnet: src (, 256, 1, 1), dst (2, 128, 1, 1)
  //
  // 2. Global shape != local shape
  //    Eg.
  //     alexnet conv5 relu
  //     src (, 384, 13, 13), dst (1, 384, 8, 13)

  // tensor in system memory
  // Global shape use local shape
  cvk_tg_t ts_data = {0};
  ts_data.base_reg_index = getTdmaBaseSelectIndexFromGaddr(ga_src);
  ts_data.fmt = tlp->fmt;
  ts_data.start_address = ga_src;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = ts_stride;

  if (do_transpose) {
    cvk_tdma_g2l_tensor_copy_nc_transposed_param_t p1 = {0};
    p1.src = &ts_data;
    p1.dst = tlp;
    tdma_g2l_tensor_copy_nc_transposed(&p1);
  } else if (do_decompress) {
    cvk_cmpr_tg_t cmpr_ts_data = {0};
    cmpr_ts_data.t = ts_data;

    cvk_tdma_g2l_tensor_copy_decompressed_param_t param = {0};
    param.src = &cmpr_ts_data;
    param.dst = tlp;
    tdma_g2l_tensor_copy_decompressed(&param);
  } else {
    cvk_tdma_g2l_tensor_copy_param_t p1 = {0};
    p1.src = &ts_data;
    p1.dst = tlp;
    tdma_g2l_tensor_copy(&p1);
  }
}

//
// Implement 1880 gdma_load, tensor format
//
// Note:
//   1880 bmk keeps eu-aligned info and calculate stride info.
//   1880v2 bmk does not keep eu-aligned info, use need to calculate stride
//   info.
//
void CviBackendContext::tdma_load(cvk_tl_t *tlp, uint64_t ga_src,
                                  uint8_t do_transpose) const {
  assert(tlp != nullptr);

  cvk_tg_t ts_data = {0};
  ts_data.fmt = tlp->fmt;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = tg_default_stride(ts_data.shape, ts_data.fmt);
  tdma_load_stride(tlp, ga_src, ts_data.stride);
}

//
// Implement 1880 gdma_store_stride, tensor format
//
// Note:
//   1880 bmk keeps eu-aligned info and calculate stride info.
//   1880v2 bmk does not keep eu-aligned info, use need to calculate stride
//   info.
//
void CviBackendContext::tdma_store_stride(cvk_tl_t *tlp, uint64_t ga_dst,
                                          cvk_tg_stride_t ts_stride,
                                          bool do_transpose,
                                          bool do_compress) const {
  assert(tlp != nullptr);

  // tensor in system memory
  // Global shape use local shape
  // Global shape used for stride calculation
  cvk_tg_t ts_data = {0};
  ts_data.base_reg_index = getTdmaBaseSelectIndexFromGaddr(ga_dst);
  ts_data.fmt = tlp->fmt;
  ts_data.start_address = ga_dst;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = ts_stride;

  if (do_transpose) {
    cvk_tdma_l2g_tensor_copy_nc_transposed_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    tdma_l2g_tensor_copy_nc_transposed(&p1);
  } else if (do_compress) {
    cvk_cmpr_tg_t cmpr_dst = {0};
    cmpr_dst.bias0 = (ts_data.fmt == CVK_FMT_BF16) ? 127 : 0;
    cmpr_dst.t = ts_data;

    cvk_tdma_l2g_tensor_copy_compressed_param_t param = {0};
    param.src = tlp;
    param.dst = &cmpr_dst;
    tdma_l2g_tensor_copy_compressed(&param);
  } else {
    cvk_tdma_l2g_tensor_copy_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    tdma_l2g_tensor_copy(&p1);
  }
}

//
// Implement 1880 gdma_store, tensor format
//
// Note:
//   1880 bmk keeps eu-aligned info and calculate stride info.
//   1880v2 bmk does not keep eu-aligned info, use need to calculate stride
//   info.
//
void CviBackendContext::tdma_store(cvk_tl_t *tlp, uint64_t ga_dst,
                                   uint8_t do_transpose) const {
  assert(tlp != nullptr);

  // tensor in system memory
  // Global shape use local shape
  // Gobal memory stride from local memory shape
  cvk_tg_t ts_data = {0};
  ts_data.base_reg_index = getTdmaBaseSelectIndexFromGaddr(ga_dst);
  ts_data.fmt = tlp->fmt;
  ts_data.start_address = ga_dst;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = tg_default_stride(ts_data.shape, ts_data.fmt);

  if (do_transpose) {
    cvk_tdma_l2g_tensor_copy_nc_transposed_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    tdma_l2g_tensor_copy_nc_transposed(&p1);
  } else {
    cvk_tdma_l2g_tensor_copy_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    tdma_l2g_tensor_copy(&p1);
  }
}

//
// Implement 1880 gdma_load_stride, matrix format
//
void CviBackendContext::tdma_load_stride(cvk_ml_t *tlp, uint64_t ga_src,
                                         cvk_mg_stride_t ts_stride,
                                         uint8_t do_transpose) const {
  assert(tlp != nullptr);

  // Global memory from reshaped local memory
  cvk_mg_t ts_data = {0};
  ts_data.base_reg_index = getTdmaBaseSelectIndexFromGaddr(ga_src);
  ts_data.start_address = ga_src;
  ts_data.fmt = tlp->fmt;
  ts_data.stride = ts_stride;

  if (do_transpose) {
    ts_data.shape = {tlp->shape.col,
                     tlp->shape.n}; // Explicit transpose shape !!!

    cvk_tdma_g2l_matrix_copy_row_col_transposed_param_t p1 = {0};
    p1.src = &ts_data;
    p1.dst = tlp;

    LLVM_DEBUG(llvm::errs() << llvm::format(
                   "tdma_load_stride(matrix): src (%d, %d), dst(n=%d, c=%d, "
                   "w=%d,col= %d)\n",
                   p1.src->shape.row, p1.src->shape.col, p1.dst->shape.n,
                   p1.dst->shape.c, p1.dst->shape.w, p1.dst->shape.col));

    tdma_g2l_matrix_copy_row_col_transposed(&p1);
  } else {
    ts_data.shape = {tlp->shape.n, tlp->shape.col};

    cvk_tdma_g2l_matrix_copy_param_t p1 = {0};
    p1.src = &ts_data;
    p1.dst = tlp;
    tdma_g2l_matrix_copy(&p1);
  }
}

//
// Implement 1880 gdma_load, matrix format
//
void CviBackendContext::tdma_load(cvk_ml_t *tlp, uint64_t ga_src,
                                  uint8_t do_transpose) const {
  assert(tlp != nullptr);
  cvk_mg_stride_t stride = {tlp->shape.col};
  if (tlp->fmt == CVK_FMT_BF16) {
    stride.row *= 2;
  }
  tdma_load_stride(tlp, ga_src, stride);
}

void CviBackendContext::tdma_load_decompress(cvk_ml_t *tlp,
                                             uint64_t ga_src) const {
  assert(tlp != nullptr);

  cvk_mg_t mg_src = {0};
  mg_src.base_reg_index = getTdmaBaseSelectIndexFromGaddr(ga_src);
  mg_src.start_address = ga_src;
  mg_src.fmt = tlp->fmt;
  mg_src.shape = {tlp->shape.n, tlp->shape.col};
  mg_src.stride = mg_default_stride(mg_src.shape, mg_src.fmt);

  cvk_cmpr_mg_t cmpr_mg_src = {0};
  cmpr_mg_src.m = mg_src;

  cvk_tdma_g2l_matrix_copy_decompressed_param_t param = {0};
  param.src = &cmpr_mg_src;
  param.dst = tlp;
  tdma_g2l_matrix_copy_decompressed(&param);
}

//
// Implement 1880 gdma_store, matrix format
//
void CviBackendContext::tdma_store(cvk_ml_t *tlp, uint64_t ga_dst,
                                   uint8_t do_transpose) const {

  assert(do_transpose == false);

  // tensor in system memory
  // Global shape use local shape
  // Gobal memory stride from local memory shape
  cvk_mg_t ts_data = {0};
  ts_data.base_reg_index = getTdmaBaseSelectIndexFromGaddr(ga_dst);
  ts_data.start_address = ga_dst;
  ts_data.fmt = tlp->fmt;
  ts_data.shape = {tlp->shape.n, tlp->shape.col};
  ts_data.stride = {tlp->shape.col};
  if (tlp->fmt == CVK_FMT_BF16) {
    ts_data.stride.row *= 2;
  }

  cvk_tdma_l2g_matrix_copy_param_t p1 = {0};
  p1.src = tlp;
  p1.dst = &ts_data;
  tdma_l2g_matrix_copy(&p1);
}

//
// Implement 1880 gdma_store_stride, matrix format
//
void CviBackendContext::tdma_store_stride(cvk_ml_t *tlp, uint64_t ga_dst,
                                          cvk_mg_stride_t ts_stride,
                                          uint8_t do_transpose) const {

  assert(do_transpose == false);

  // tensor in system memory
  // Global shape use local shape
  // Global shape used for stride calculation
  cvk_mg_t ts_data = {0};
  ts_data.base_reg_index = getTdmaBaseSelectIndexFromGaddr(ga_dst);
  ts_data.start_address = ga_dst;
  ts_data.fmt = tlp->fmt;
  ts_data.shape = {tlp->shape.n, tlp->shape.col};
  ts_data.stride = ts_stride;

  cvk_tdma_l2g_matrix_copy_param_t p1 = {0};
  p1.src = tlp;
  p1.dst = &ts_data;
  tdma_l2g_matrix_copy(&p1);
}

void CviBackendContext::tdma_g2g_tensor_copy(
    uint64_t src_addr, cvk_tg_shape_t src_shape, cvk_tg_stride_t src_stride,
    cvk_fmt_t src_fmt, uint64_t dst_addr, cvk_tg_shape_t dst_shape,
    cvk_tg_stride_t dst_stride, cvk_fmt_t dst_fmt) const {
  cvk_tg_t src = {0};
  src.start_address = src_addr;
  src.base_reg_index = getTdmaBaseSelectIndexFromGaddr(src_addr);
  src.fmt = src_fmt;
  src.shape = src_shape;
  src.stride = src_stride;
  src.int8_rnd_mode = 0;

  cvk_tg_t dst = {0};
  dst.start_address = dst_addr;
  dst.base_reg_index = getTdmaBaseSelectIndexFromGaddr(dst_addr);
  dst.fmt = dst_fmt;
  dst.shape = dst_shape;
  dst.stride = dst_stride;
  dst.int8_rnd_mode = 0;
  cvk_tdma_g2g_tensor_copy_param_t p = {0};
  p.src = &src;
  p.dst = &dst;
  tdma_g2g_tensor_copy(&p);
}

int CviBackendContext::bitsize_of_fmt(uint32_t fmt) const {
  switch (fmt) {
  case CVK_FMT_F32:
  case CVK_FMT_I32:
    return 32;
  case CVK_FMT_BF16:
  case CVK_FMT_F16:
  case CVK_FMT_I16:
  case CVK_FMT_U16:
    return 16;
  case CVK_FMT_I8:
  case CVK_FMT_U8:
    return 8;
  case CVK_FMT_I4:
    return 4;
  case CVK_FMT_I2:
    return 2;
  case CVK_FMT_I1:
    return 1;
  default:
    assert(0);
    return -1;
  }
}

const cvk_tl_shape_t &CviBackendContext::lut_table_shape(cvk_fmt_t fmt) const {
  static const cvk_tl_shape_t table_fixed = tl_shape_t4(1, NPU_NUM, 16, 16);
  static const cvk_tl_shape_t table_bf16 = tl_shape_t4(1, NPU_NUM, 32, 8);
  assert_support_fmt(fmt);
  if (fmt == CVK_FMT_BF16) {
    return table_bf16;
  }
  return table_fixed;
}

bool CviBackendContext::size_to_hw(int size, int&h, int&w) const {
  if (size <= MAX_WIDTH) {
    h = 1;
    w = size;
    return true;
  }
  int div = std::sqrt(size);
  for (h = div; h >= 2; h--) {
    if (size % h == 0) {
      w = size / h;
      break;
    }
  }
  if (h <= MAX_WIDTH && w <= MAX_WIDTH) {
    return true;
  }
  return false;
}

void CviBackendContext::tiling_all(std::vector<tiling_info_t> &tiling_result,
                                   int64_t total, cvk_fmt_t fmt, int blob_num,
                                   uint32_t lmem_size) const {
  tiling_info_t tile;
  memset(&tile, 0, sizeof(tile));
  tile.n = 1;
  tile.c = NPU_NUM;
  tile.w = tiu_eu_num(fmt);
  tile.h = std::min(ceiling_func(total, tile.c * tile.w), MAX_HEIGHT);
  bool lmem_ok = false;
  while (total > 0) {
    int64_t count = tile.n * tile.c * tile.h * tile.w;
    if (lmem_ok == false) {
      uint32_t lsize = blob_num * lmem_tensor_to_size(tile.n, tile.c, tile.h,
                                                      tile.w, fmt, 1);
      lmem_ok = (lsize <= lmem_size);
    }
    if (count > total || lmem_ok == false) {
      if (tile.h > 1) {
        tile.h--;
      } else if (tile.w > 1) {
        tile.w--;
      } else if (tile.c > 1) {
        tile.c--;
      } else {
        assert(0 && "lmem is not enough");
      }
    } else {
      LLVM_DEBUG(llvm::errs() << llvm::format(
                     "Tiles all, tile:(%d,%d,%d,%d), offset:%lu\n", tile.n,
                     tile.c, tile.h, tile.w, tile.offset););
      tiling_result.emplace_back(tile);
      total -= count;
      tile.offset += count * bytesize_of_fmt(fmt);
    }
  }
  assert(total == 0 && "tiling error");
  return;
}

void CviBackendContext::tiling_nchw(std::vector<tiling_info_t> &tiling_result,
                                    int n, int c, int h, int w, cvk_fmt_t fmt,
                                    int blob_num, uint32_t lmem_size,
                                    tiling_mode_t mode) const {
  int max_w = std::min(w, MAX_WIDTH);
  int max_h = std::min(h, MAX_HEIGHT);
  int max_c = std::min(c, MAX_CHANNEL);
  int max_n = std::min(n, MAX_CHANNEL);
  int min_c = 1;
  int min_w = 1;
  if (mode == TilingNHW) { // keep c
    assert(max_c == c && "keep c, but c too large");
    min_c = max_c;
  } else if (mode == TilingNCH) { // keep w
    assert(max_w == w && "keep w, but w too large");
    min_w = max_w;
  } else if (mode == TilingNH) { // keep cw
    assert(max_c == c && "keep c, but c too large");
    assert(max_w == w && "keep w, but w too large");
    min_c = max_c;
    min_w = max_w;
  }


  int step_w, step_h, step_c, step_n;
  uint32_t lmem_required = 0;
  for (step_w = max_w; step_w >= min_w; --step_w) {
    for (step_h = max_h; step_h >= 1; --step_h) {
      for (step_n = max_n; step_n >= 1; --step_n) {
        for (step_c = max_c; step_c >= min_c;) {
          cvk_tl_shape_t max_shape =
              tl_shape_t4(step_n, step_c, step_h, step_w);
          lmem_required = blob_num * lmem_tensor_to_size(max_shape, fmt, 1);
          if (lmem_required <= lmem_size) {
            goto after_loop;
          }
          if (step_c % NPU_NUM) {
            step_c -= step_c % NPU_NUM;
          } else {
            step_c -= NPU_NUM;
          }
        }
      }
    }
  }
after_loop:
  if (lmem_required > lmem_size) {
    llvm::errs() << llvm::format(
        "Tilling failed, src shape:(%d,%d,%d,%d), fmt:%d\n", n, c, h, w, fmt);
    assert(0);
  }

  tiling_info_t tile;
  cvk_tg_stride_t src_stride = tg_default_stride(c, h, w, fmt);
  for (tile.pos_n = 0; tile.pos_n < n; tile.pos_n += step_n) {
    tile.n = std::min(n - tile.pos_n, step_n);
    for (tile.pos_c = 0; tile.pos_c < c; tile.pos_c += step_c) {
      tile.c = std::min(c - tile.pos_c, step_c);
      for (tile.pos_h = 0; tile.pos_h < h; tile.pos_h += step_h) {
        tile.h = std::min(h - tile.pos_h, step_h);
        for (tile.pos_w = 0; tile.pos_w < w; tile.pos_w += step_w) {
          tile.w = std::min(w - tile.pos_w, step_w);
          tile.offset = tile.pos_w * src_stride.w + tile.pos_h * src_stride.h +
                        tile.pos_c * src_stride.c + tile.pos_n * src_stride.n;
          tiling_result.emplace_back(tile);
        }
      }
    }
  }
  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "Tiles, mode:%d, shape:(%d,%d,%d,%d), step:(%d,%d,%d,%d)\n",
                 mode, n, c, h, w, step_n, step_c, step_h, step_w););
}

void CviBackendContext::tiling_packing(
    std::vector<tiling_info_t> &tiling_result, cvk_tg_shape_t shape,
    cvk_fmt_t fmt, int blob_num, uint32_t reserved_lmem,
    tiling_mode_t mode) const {
  int n = static_cast<int>(shape.n);
  int c = static_cast<int>(shape.c);
  int h = static_cast<int>(shape.h);
  int w = static_cast<int>(shape.w);
  tiling_packing(tiling_result, n, c, h, w, fmt, blob_num, reserved_lmem, mode);
}

void CviBackendContext::tiling_packing(
    std::vector<tiling_info_t> &tiling_result, int n, int c, int h, int w,
    cvk_fmt_t fmt, int blob_num, uint32_t reserved_lmem,
    tiling_mode_t mode) const {
  uint32_t lmem_size = (uint32_t)LOCAL_MEM_SIZE - reserved_lmem;
  assert((uint32_t)LOCAL_MEM_SIZE > reserved_lmem && "reserved_lmem too large");

  if (mode == TilingAll) {
    tiling_all(tiling_result, n * c * h * w, fmt, blob_num, lmem_size);
  } else {
    tiling_nchw(tiling_result, n, c, h, w, fmt, blob_num, lmem_size, mode);
  }
}

void CviBackendContext::assert_support_fmt(cvk_fmt_t fmt) const {
  assert((fmt == CVK_FMT_I8 || fmt == CVK_FMT_U8 || fmt == CVK_FMT_BF16) &&
         "others not supported");
}

void *cvi_backend_get_cvk_ctx(const CviBackendContext &ctx) {
  return ctx.get_cvk_ctx();
}

uint32_t CviBackendContext::ga_cmpr_offset(int n, int c, int h, int w,
                                           int n_pos, int c_pos, int h_pos,
                                           int c_step, int step_size) const {
  uint32_t cmpr_n_offset = n_pos * llvm::divideCeil(c, c_step) * h * step_size;
  uint32_t cmpr_c_offset = (c_pos / c_step) * h * step_size;
  uint32_t cmpr_h_offset = h_pos * step_size;

  return cmpr_n_offset + cmpr_c_offset + cmpr_h_offset;
}

uint32_t CviBackendContext::addr_after_right_shift(int addr, uint32_t step,
                                                   int c_str) const {
  uint32_t lmem_i = (addr / LOCAL_MEM_SIZE + step) % NPU_NUM;
  uint32_t offset = addr % LOCAL_MEM_SIZE + (lmem_i + step) / NPU_NUM * c_str;
  return lmem_i * LOCAL_MEM_SIZE + offset;
}

uint32_t CviBackendContext::tl_cmpr_c_stride(int n, int c, int h, int w,
                                             cvk_fmt_t fmt) const {
  // (1, c, h, w) -> (1, NPU, 1, w) ...
  // Right shift NPU, same as next batch so eu_align = 1
  cvk_tl_shape_t blck_cmpr_shape = tl_shape_t4(n, c, h, w);
  cvk_tl_stride_t blck_cmpr_stride = tl_default_stride(blck_cmpr_shape, fmt, 1);

  return blck_cmpr_stride.c;
}

void CviBackendContext::tiu_zeros(uint16_t layer_id, cvk_tl_t *tl_mem) const {
  assert(tl_mem);
  if (tl_mem->fmt == CVK_FMT_BF16) {
    auto stride =
        tl_default_stride(tl_mem->shape, tl_mem->fmt, tl_mem->eu_align);
    // try to use xor first
    if (0 == memcmp(&stride, &tl_mem->stride, sizeof(stride))) {
      cvk_tl_t tl_z = *tl_mem;
      tl_z.fmt = CVK_FMT_I8;
      tl_z.cmprs_fmt = tl_z.fmt;
      tl_z.shape.w *= 2;
      tl_z.stride = tl_default_stride(tl_z.shape, tl_z.fmt, tl_z.eu_align);
      cvk_tiu_xor_int8_param_t p = {0};
      p.res = &tl_z;
      p.a = &tl_z;
      p.b = &tl_z;
      p.layer_id = layer_id;
      tiu_xor_int8(&p);
    } else {
      cvk_tiu_sub_param_t p = {0};
      p.res_low = tl_mem;
      p.a_low = tl_mem;
      p.b_low = tl_mem;
      p.rshift_bits = 0;
      p.layer_id = layer_id;
      tiu_sub(&p);
    }
  } else {
    cvk_tiu_xor_int8_param_t p = {0};
    p.res = tl_mem;
    p.a = tl_mem;
    p.b = tl_mem;
    p.layer_id = layer_id;
    tiu_xor_int8(&p);
  }
}

template <typename FPtrTy> FPtrTy CV18xx::CastToFPtr(const char *symbolName) {
  assert(DL.isValid());
  auto fPtr = DL.getAddressOfSymbol(symbolName);
  if (fPtr == nullptr) {
    llvm::errs() << "can't find symbol: " << symbolName << "\n";
    llvm_unreachable(symbolName);
  }
  return reinterpret_cast<FPtrTy>(fPtr);
}

#define CAST_FUNCTION(name) dl_##name = CastToFPtr<name>(#name)

void CV18xx::init() {
    if (!DL.isValid()) {
    std::string Err;
    DL = llvm::sys::DynamicLibrary::getPermanentLibrary(LIB_NAME.data(), &Err);
    CAST_FUNCTION(cvikernel_register);
    if (DL.isValid() == false) {
      llvm_unreachable(Err.c_str());
    }
  }
}

cvk_fmt_t CV18xx::getDataType(Value v) {
  auto type = Module::getStorageType(v);
  return getDataType(type);
}

cvk_fmt_t CV18xx::getDataType(mlir::Type type) {
  auto bits = type.getIntOrFloatBitWidth();
  if (bits == 8) {
    if (type.isUnsignedInteger()) {
      return CVK_FMT_U8;
    } else {
      return CVK_FMT_I8;
    }
  } else if (type.isF32()) {
    return CVK_FMT_F32;
  } else if (type.isBF16()) {
    return CVK_FMT_BF16;
  }
  type.dump();
  llvm_unreachable("Unsupport type \n");
  return CVK_FMT_F32;
}
} // namespace backend
} // namespace tpu_mlir

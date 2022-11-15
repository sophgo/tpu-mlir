//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Support/TPUCompressUtil.h"
#define DEBUG_TYPE "crop_kernel"

namespace tpu_mlir {
namespace backend {
static void tg_crop_with_w_step(const CviBackendContext &ctx, uint32_t layer_id,
                                gaddr_t ga_input, gaddr_t ga_output,
                                std::vector<int64_t> is_4,
                                std::vector<int64_t> os_4,
                                std::vector<int> offsets,
                                std::vector<int> steps, cvk_fmt_t fmt) {
  ctx.set_layer_id(layer_id);
  auto i_gstride = ctx.tg_default_stride(is_4[1], is_4[2], is_4[3], fmt);
  auto i_gstride2 = i_gstride;
  i_gstride2.n *= steps[0];
  i_gstride2.c *= steps[1];
  i_gstride2.h *= steps[2];
  auto o_s = ctx.tg_shape_t4(os_4[0], os_4[1], os_4[2], os_4[3]);
  auto o_gstride = ctx.tg_default_stride(o_s, fmt);
  std::vector<CviBackendContext::tiling_info_t> tiles;
  int num_blobs = steps[3] + 1;
  ctx.tiling_packing(tiles, o_s, fmt, num_blobs);
  for (auto &tile : tiles) {
    auto in = offsets[0] + tile.pos_n * steps[0];
    auto ic = offsets[1] + tile.pos_c * steps[1];
    auto ih = offsets[2] + tile.pos_h * steps[2];
    auto iw = offsets[3] + tile.pos_w * steps[3];
    uint64_t input_offset = in * i_gstride.n + ic * i_gstride.c +
                            ih * i_gstride.h + iw * i_gstride.w;
    auto ishape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w * steps[3]);
    auto *tl_input = ctx.lmem_alloc_tensor(ishape, fmt, 1);
    ctx.tdma_load_stride(tl_input, ga_input + input_offset, i_gstride2);
    auto oshape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    auto *tl_output = ctx.lmem_alloc_tensor(oshape, fmt, 1);
    tl_input->shape.w = tile.w;
    tl_input->stride.w *= steps[3];
    cvk_tiu_copy_param_t p = {0};
    p.src = tl_input;
    p.dst = tl_output;
    p.layer_id = layer_id;
    ctx.tiu_copy(&p);
    ctx.tdma_store_stride(tl_output, ga_output + tile.offset, o_gstride);
    ctx.lmem_free_tensor(tl_output);
    ctx.lmem_free_tensor(tl_input);
  }
}

void cvi_backend_tg_crop_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                                gaddr_t ga_input, gaddr_t ga_output,
                                std::vector<int64_t> is_4,
                                std::vector<int64_t> os_4,
                                std::vector<int> offsets,
                                std::vector<int> steps, cvk_fmt_t fmt) {
  if (steps[3] != 1) {
    tg_crop_with_w_step(ctx, layer_id, ga_input, ga_output, is_4, os_4, offsets,
                        steps, fmt);
    return;
  }
  ctx.set_layer_id(layer_id);
  auto i_gstride = ctx.tg_default_stride(is_4[1], is_4[2], is_4[3], fmt);
  auto i_gstride2 = i_gstride;
  i_gstride2.n *= steps[0];
  i_gstride2.c *= steps[1];
  i_gstride2.h *= steps[2];
  auto o_s = ctx.tg_shape_t4(os_4[0], os_4[1], os_4[2], os_4[3]);
  auto o_gstride = ctx.tg_default_stride(o_s, fmt);

  std::vector<CviBackendContext::tiling_info_t> tiles;
  ctx.tiling_packing(tiles, o_s, fmt);
  for (auto &tile : tiles) {
    auto in = offsets[0] + tile.pos_n * steps[0];
    auto ic = offsets[1] + tile.pos_c * steps[1];
    auto ih = offsets[2] + tile.pos_h * steps[2];
    auto iw = offsets[3] + tile.pos_w * steps[3];
    uint64_t input_offset = in * i_gstride.n + ic * i_gstride.c +
                            ih * i_gstride.h + iw * i_gstride.w;
    auto ishape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    cvk_tl_t *tl_ifmap = ctx.lmem_alloc_tensor(ishape, fmt, 1);
    ctx.tdma_load_stride(tl_ifmap, ga_input + input_offset, i_gstride2);
    ctx.tdma_store_stride(tl_ifmap, ga_output + tile.offset, o_gstride);
    ctx.lmem_free_tensor(tl_ifmap);
  }
}
} // namespace backend
} // namespace tpu_mlir

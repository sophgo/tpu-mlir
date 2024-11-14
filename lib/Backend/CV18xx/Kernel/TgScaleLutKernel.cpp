//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Backend/CV18xx/Kernel/TgScaleLutKernel.hpp"

#define DEBUG_TYPE "cvi_backend_scalelut_kernel"
namespace tpu_mlir {
namespace backend {
void TgScaleLutKernel::init(uint32_t layer_id, gaddr_t ga_input,
                            gaddr_t ga_output, gaddr_t ga_lut, int n, int c,
                            int h, int w, cvk_fmt_t fmt) {
  this->layer_id = layer_id;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  this->c_times = 1;
  this->c_ori = c;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->ga_lut = ga_lut;
  this->fmt = fmt;
  reshape();
  lut_shape = CV18xx::lut_table_shape(fmt);
  gstride = CV18xx::tg_default_stride(this->c, this->h, this->w, fmt);
  CV18xx::set_layer_id(layer_id);
}

#define MAX_H_STRIDE 0x10000
void TgScaleLutKernel::reshape() {
  if (CV18xx::NPU_NUM < c_ori) {
    // if CV18xx::NPU_NUM < c, put all lut table in each NPU
    // 1 3 540 960 => {3, CV18xx::NPU_NUM, 1, 540 * 960 / CV18xx::NPU_NUM}
    int _n = this->n;
    int _c = this->c;
    int _h = this->h;
    int _w = this->w;

    _n *= _c;
    _w *= _h;
    _h = 1;

    for (int i = CV18xx::NPU_NUM; i > 0; --i) {
      if (_w % CV18xx::NPU_NUM == 0) {
        _c = i;
        _w = _w / CV18xx::NPU_NUM;
        break;
      }
    }
    LLVM_DEBUG(llvm::dbgs()
               << llvm::format("reshape [%d,%d,%d,%d] => [%d,%d,%d,%d]\n", n, c,
                               h, w, _n, _c, _h, _w));
    this->n = _n;
    this->c = _c;
    this->h = _h;
    this->w = _w;
    tl_lut.resize(c_ori);
  } else {
    // for example
    // 1 3 540 960 => 1 30 1 54*960 = 1 30 180 288
    int c_ = c;
    int h_ = 1;
    int w_ = h * w;
    int max_times = CV18xx::NPU_NUM / c;
    for (int time = max_times; time >= 2; time--) {
      if (w_ % time == 0) {
        w_ /= time;
        c_ *= time;
        break;
      }
    }
    if (c == c_) {
      tl_lut.resize(1);
      return;
    }
    if (w_ > MAX_WIDTH) {
      int div = std::sqrt(w_);
      for (int time = div; time >= 2; time--) {
        if (w_ % time == 0) {
          w_ /= time;
          h_ = time;
          break;
        }
      }
      if (w_ >= MAX_H_STRIDE && h_ >= MAX_H_STRIDE) {
        assert(0 && "w_ >= MAX_H_STRIDE && h_ >= MAX_H_STRIDE");
        return;
      }
      if (w_ >= MAX_H_STRIDE) {
        std::swap(w_, h_);
      } else if (h_ > w_ && h_ < MAX_H_STRIDE) {
        std::swap(w_, h_);
      }
    }
    LLVM_DEBUG(llvm::dbgs()
               << llvm::format("reshape [%d,%d,%d,%d] => [%d,%d,%d,%d]\n", n, c,
                               h, w, n, c_, h_, w_));
    c_times = c_ / c;
    c = c_;
    h = h_;
    w = w_;
    tl_lut.resize(1);
  }
}

void TgScaleLutKernel::selectTilePolicy() {
  uint32_t lmem_used = CV18xx::lmem_tensor_to_size(lut_shape, fmt, 1);
  if (CV18xx::NPU_NUM < c_ori) {
    int w_step = w;
    for (w_step = w; w_step > 0; --w_step) {
      cvk_tl_shape_t data_shape = {(uint32_t)n, (uint32_t)c, (uint32_t)h,
                                   (uint32_t)w_step};
      uint32_t need_mem_size = CV18xx::lmem_tensor_to_size(data_shape, fmt, 1);
      if (need_mem_size * BLOB_NUM <=
          (uint32_t)CV18xx::LMEM_BYTES - c_ori * lmem_used) {
        break;
      }
    }
    if (w_step > 0) {
      for (int w_loc = 0; w_loc < w;) {
        CV18xx::tiling_info_t tile;
        tile.n = n;
        tile.c = c;
        tile.h = h;
        tile.w = std::min(w_step, w - w_loc);
        tile.pos_n = 0;
        tile.pos_c = 0;
        tile.pos_h = 0;
        tile.pos_w = w_loc;
        tile.offset = w_loc;
        w_loc += tile.w;
        tiles.emplace_back(tile);
      }
    } else {
      assert(0 && "TgScaleLutKernel fail!\n");
    }
  } else {
    CV18xx::tiling_packing(tiles, n, c, h, w, fmt, BLOB_NUM, lmem_used,
                           CV18xx::TilingNHW);
  }
}

void TgScaleLutKernel::allocLmem() {
  cvk_tl_shape_t gshape =
      CV18xx::tl_shape_t4(tiles[0].n, tiles[0].c, tiles[0].h, tiles[0].w);
  if (CV18xx::NPU_NUM < c_ori) {
    for (int i = 0; i < c_ori; ++i) {
      tl_lut[i] = CV18xx::lmem_alloc_tensor(lut_shape, fmt, 1);
    }
  } else {
    tl_lut[0] = CV18xx::lmem_alloc_tensor(lut_shape, fmt, 1);
  }
  for (int i = 0; i < BLOB_NUM; i++) {
    tl_mem[i] = CV18xx::lmem_alloc_tensor(gshape, fmt, 1);
  }
  // load table
  if (CV18xx::NPU_NUM < c_ori) {
    for (int i = 0; i < c_ori; ++i) {
      int hw = lut_shape.h * lut_shape.w;
      tl_lut[i]->shape = CV18xx::tl_shape_t4(1, CV18xx::NPU_NUM, 1, hw);
      tl_lut[i]->stride = CV18xx::tl_default_stride(tl_lut[i]->shape, fmt, 1);
      cvk_tg_t ts_data = {0};
      ts_data.base_reg_index = CV18xx::getTdmaBaseSelectIndexFromGaddr(ga_lut);
      ts_data.fmt = fmt;
      ts_data.start_address = ga_lut + i * tl_lut[i]->stride.c;
      ts_data.shape = CV18xx::tg_shape_t4(1, CV18xx::NPU_NUM, 1, hw);
      ts_data.stride = {(uint32_t)(CV18xx::NPU_NUM * hw), 0,
                        (uint32_t)lut_shape.w, 1};
      cvk_tdma_g2l_tensor_copy_param_t p = {0};
      p.src = &ts_data;
      p.dst = tl_lut[i];
      p.layer_id = layer_id;
      CV18xx::tdma_g2l_tensor_copy(&p);
      tl_lut[i]->shape = lut_shape;
      tl_lut[i]->stride = CV18xx::tl_default_stride(lut_shape, fmt, 1);
    }
  } else {
    int hw = lut_shape.h * lut_shape.w;
    cvk_tg_t ts_data = {0};
    ts_data.base_reg_index = CV18xx::getTdmaBaseSelectIndexFromGaddr(ga_lut);
    ts_data.fmt = fmt;
    ts_data.start_address = ga_lut;
    ts_data.shape = CV18xx::tg_shape_t4(1, c / c_times, c_times, hw);
    ts_data.stride = {(uint32_t)(c / c_times * hw), (uint32_t)hw, 0, 1};
    tl_lut[0]->shape = CV18xx::tl_shape_t4(1, c, 1, hw);
    tl_lut[0]->stride = CV18xx::tl_default_stride(tl_lut[0]->shape, fmt, 1);
    cvk_tdma_g2l_tensor_copy_param_t p = {0};
    p.src = &ts_data;
    p.dst = tl_lut[0];
    p.layer_id = layer_id;
    CV18xx::tdma_g2l_tensor_copy(&p);
    tl_lut[0]->shape = lut_shape;
    tl_lut[0]->stride = CV18xx::tl_default_stride(lut_shape, fmt, 1);
  }
}

void TgScaleLutKernel::deallocLmem() {
  for (int i = BLOB_NUM - 1; i >= 0; i--) {
    CV18xx::lmem_free_tensor(tl_mem[i]);
  }
  if (CV18xx::NPU_NUM < c_ori) {
    for (int i = c_ori - 1; i >= 0; i--) {
      CV18xx::lmem_free_tensor(tl_lut[i]);
    }
  } else {
    CV18xx::lmem_free_tensor(tl_lut[0]);
  }
}

void TgScaleLutKernel::schedule() {
  allocLmem();
  int32_t total_steps = tiles.size();
  for (int32_t i = 0; i < total_steps + 2; i++) {
    CV18xx::parallel_enable();

    if (i - 1 >= 0 && i - 1 < total_steps) {
      compute(i - 1);
    }
    if (i < total_steps) {
      load(i);
    }
    if (i - 2 >= 0) {
      store(i - 2);
    }
    CV18xx::parallel_disable();
  }
  deallocLmem();
}

void TgScaleLutKernel::refresh(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  tl_ifmap = *tl_mem[step_idx % 2];
  tl_ofmap = *tl_mem[2 + step_idx % 2];
  auto shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  auto stride = CV18xx::tl_default_stride(shape, fmt, 1);
  tl_ifmap.shape = shape;
  tl_ifmap.stride = stride;
  tl_ofmap.shape = shape;
  tl_ofmap.stride = stride;
}

void TgScaleLutKernel::load(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  CV18xx::tdma_load_stride(&tl_ifmap, ga_input + tile.offset, gstride);
}

void TgScaleLutKernel::store(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  CV18xx::tdma_store_stride(&tl_ofmap, ga_output + tile.offset, gstride);
}

void TgScaleLutKernel::compute(int32_t step_idx) {
  refresh(step_idx);
  if (CV18xx::NPU_NUM < c_ori) {
    for (int i = 0; i < c_ori; ++i) {
      auto tl_cur_ifmap = tl_ifmap;
      tl_cur_ifmap.shape.n = n / c_ori;
      tl_cur_ifmap.stride.n *= c_ori;
      tl_cur_ifmap.start_address =
          tl_ifmap.start_address + i * tl_ifmap.stride.n;

      auto tl_cur_ofmap = tl_ofmap;
      tl_cur_ofmap.shape.n = n / c_ori;
      tl_cur_ofmap.stride.n *= c_ori;
      tl_cur_ofmap.start_address =
          tl_ofmap.start_address + i * tl_ofmap.stride.n;
      cvk_tiu_lookup_table_param_t p = {0};
      p.ofmap = &tl_cur_ofmap;
      p.ifmap = &tl_cur_ifmap;
      p.table = tl_lut[i];
      p.layer_id = layer_id;
      CV18xx::tiu_lookup_table(&p);
    }
  } else {
    cvk_tiu_lookup_table_param_t p = {0};
    p.ofmap = &tl_ofmap;
    p.ifmap = &tl_ifmap;
    p.table = tl_lut[0];
    p.layer_id = layer_id;
    CV18xx::tiu_lookup_table(&p);
  }
}

void cvi_backend_tg_scale_lut_kernel(uint32_t layer_id, gaddr_t ga_input,
                                     gaddr_t ga_output, gaddr_t ga_lut, int n,
                                     int c, int h, int w, cvk_fmt_t fmt) {
  TgScaleLutKernel kernel;
  kernel.init(layer_id, ga_input, ga_output, ga_lut, n, c, h, w, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}
} // namespace backend
} // namespace tpu_mlir

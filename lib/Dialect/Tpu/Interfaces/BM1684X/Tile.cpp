//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"




using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int tile_coeff[MAX_SHAPE_DIMS];
    int type;
} tile_common_spec_t;

typedef struct {
    tile_common_spec_t common;
    uint64_t buffer_addr;
} tile_global_spec_t;

typedef struct {
    tile_global_spec_t spec;
    int coeff_is_fixed;
    int input_is_coeff;
    int input_shape[MAX_SHAPE_DIMS];
    int input_dims;
    int dtype;
} tile_global_param_t;

typedef struct {
  int                tile_axis;
  int                tile_num;
  int                type;
} tile_1d_global_param_t;

typedef struct {
    tile_common_spec_t common;
} tile_local_spec_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GloballGenInterface
// =========================================
void tpu::TileOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto& in = input_spec->at(0);
  auto& out = output_spec->at(0);
  if (in.dims < out.dims) {
    const int more_dims = out.dims - in.dims;
    memmove(in.shape + more_dims, in.shape, in.dims * sizeof(int));
    for (int i = 0; i < more_dims; ++i) {
      in.shape[i] = 1;
    }
    in.dims = out.dims;
  }
  tile_global_spec_t spec = {0};
  for (int i = 0; i < out.dims; ++i) {
    spec.common.tile_coeff[i] = out.shape[i] / in.shape[i];
  }
  spec.common.type = 0;
  BM168x::call_global_func("backend_api_tile_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::TileOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                           int64_t out_lmem_bytes,
                                           int64_t in_nslice, int64_t in_hslice,
                                           int64_t out_nslice,
                                           int64_t out_hslice) {
  return 0;
}

void tpu::TileOp::assign_sec_info(int64_t n_step, int64_t h_step,
                                  void *sec_info_) {
  local_sec_info_t *sec_info = (local_sec_info_t *)sec_info_;
  memset(sec_info, 0, sizeof(local_sec_info_t));

  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  sec_info->n_slice = in_gi.n_slice;
  sec_info->d_slice = 1;
  sec_info->h_slice = in_gi.h_slice;
  sec_info->h_idx = in_gi.h_idx;
  sec_info->is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == h);
  sec_info->w_slice = w;
  sec_info->out_n_slice = gi.n_slice;
  sec_info->out_h_idx = gi.h_idx;
  sec_info->out_h_slice = gi.h_slice;
  sec_info->out_w_slice = w;
}

void tpu::TileOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                        void *sec_info_) {
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto& in = input_spec->at(0);
  auto& out = output_spec->at(0);
  tile_local_spec_t spec = {0};
  for (int i = 0; i < in.dims; ++i) {
    spec.common.tile_coeff[i] = out.shape[i] / in.shape[i];
  }
  spec.common.type = 0;

  BM168x::call_local_func("backend_api_tile_local", &spec, sizeof(spec),
                          sec_info_, input_spec->data(), output_spec->data());
}

//dynamic codegen
int64_t tpu::TileOp::dyn_codegen_local_bm1684x(void *buffer) {
return 0;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::TileOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

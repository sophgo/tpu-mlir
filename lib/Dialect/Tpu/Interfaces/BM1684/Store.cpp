//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::StoreOp::codegen_global_bm1684() {
  UNREACHABLE_THIS("Not Implemented");
}

int64_t tpu::StoreOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                           int64_t out_lmem_bytes,
                                           int64_t in_nslice, int64_t in_hslice,
                                           int64_t out_nslice,
                                           int64_t out_hslice) {
  return 0;
}

void tpu::StoreOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                        local_sec_info_t &sec_info) {
  CMD_ID_NODE *pid_node = (CMD_ID_NODE *)(*BM168x::instance())->gdma_node;
  auto gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto data_type = BM168x::getDataType(getOutput());
  // FP32 INT8 INT16 are all move as FLOAT32 because of 2N 4N store mode
  auto gdma_format = BM168x::GDMA_VALUE_FORMAT_FLOAT32;
  auto force_1N_store_format = BM168x::GDMA_VALUE_FORMAT_FLOAT32;
  int64_t fmt_bytes = (int64_t)BM168x::getFmtBytes(DTYPE_FP32);
  int64_t N, C, D, H, W;
  module::getNCDHW(getOutput(), N, C, D, H, W, (group_type_t)gi.type);
  int64_t n_idx = gi.n_idx;
  int64_t local_N = gi.n_slice, local_C = C, local_H = gi.h_slice, local_W = W;
  int64_t c_idx_in = 0;
  int64_t c_num_local = ceiling_func(C, BM168x::NPU_NUM);
  int64_t c_stride = gi.eu_align
                         ? align_up(local_H * local_W, Arch::eu_num(fmt_bytes))
                         : local_H * local_W;
  int64_t h_stride = local_W;
  int n_idx_trans, n_slice_trans, data_type_size;
  switch (data_type) {
  case DTYPE_INT8:
  case DTYPE_UINT8:
    n_idx_trans = n_idx / 4;
    n_slice_trans = ceiling_func(local_N, 4);
    data_type_size = 1;
    force_1N_store_format = BM168x::GDMA_VALUE_FORMAT_UINT8;
    break;
  case DTYPE_INT16:
  case DTYPE_UINT16:
  case DTYPE_FP16:
    n_idx_trans = n_idx / 2;
    n_slice_trans = ceiling_func(local_N, 2);
    data_type_size = 2;
    force_1N_store_format = BM168x::GDMA_VALUE_FORMAT_FLOAT16;
    break;
  default:
    n_idx_trans = n_idx;
    n_slice_trans = local_N;
    data_type_size = 4;
    force_1N_store_format = BM168x::GDMA_VALUE_FORMAT_FLOAT32;
    break;
  }
  auto g_addr = module::getAddress(getOutput());
  auto float_size = BM168x::getFmtBytes(DTYPE_FP32);
  for (int d_idx = 0; d_idx < D; d_idx++) {
    int cur_local_offset =
        d_idx * n_slice_trans * c_num_local * c_stride * float_size;
    if ((int64_t)BM168x::getFmtBytes(data_type) != 4) {
      c_num_local = ceiling_func(D * local_C, BM168x::NPU_NUM);
      cur_local_offset = ((c_idx_in + d_idx * local_C) % BM168x::NPU_NUM) *
                             BM168x::LMEM_BYTES +
                         (c_idx_in + d_idx * local_C) / BM168x::NPU_NUM *
                             c_stride * float_size;
    }
    int64_t cur_global_offset =
        float_size * (n_idx_trans * C * D * H * W + gi.c_idx * D * H * W +
                      d_idx * H * W + gi.h_idx * W + gi.w_idx);
    BM168x::instance()->dl_tensor_stride_move_gen_cmd(
        gi.out_addr + cur_local_offset, 0, g_addr + cur_global_offset,
        n_slice_trans, local_C, local_H, local_W, c_num_local * c_stride,
        c_stride, h_stride, 1, C * D * H * W, D * H * W, W, 1, gdma_format,
        GDMA_VALUE_DIR_L2S, // 1,
        0, pid_node);
  }
}

uint32_t tpu::StoreOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::StoreOp::get_fw_type_bm1684() { return -1; }

int32_t tpu::StoreOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

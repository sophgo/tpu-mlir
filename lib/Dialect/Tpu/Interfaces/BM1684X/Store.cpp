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
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include <algorithm>
using namespace tpu_mlir::backend;

void tpu::StoreOp::codegen_global_bm1684x() {
  llvm_unreachable("Not Implemented");
}

int64_t tpu::StoreOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return 0;
}

void tpu::StoreOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                         int64_t h_step, int64_t d_step,
                                         int64_t w_step,
                                         group_type_t group_type,
                                         local_sec_info_t &sec_info) {
  CMD_ID_NODE *pid_node = (CMD_ID_NODE *)(*BM168x::instance())->gdma_node;
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  int64_t N, C, D, H, W, real_hslice, real_wslice, real_dslice, real_cslice;
  int64_t gdma_format;
  auto shape = module::getShape(getInput());
  auto data_type = BM168x::getDataType(getOutput());
  real_hslice = gi.h_slice;
  real_wslice = gi.w_slice;
  real_dslice = gi.d_slice;
  real_cslice = gi.c_slice;
  if (data_type == DTYPE_UINT4 || data_type == DTYPE_INT4) {
    gdma_format = BM168x::GDMA_VALUE_FORMAT_INT8;
    data_type = DTYPE_INT8;
    if (shape.size() == 2) {
      N = 1;
      C = shape[0];
      D = 1;
      H = 1;
      W = align_up(shape[1], (int64_t)2) / 2;
      real_wslice = align_up(shape[1], (int64_t)2) / 2;
    } else {
      module::getNCDHW(getOutput(), N, C, D, H, W, group_type);
      if (gi.h_slice == H) {
        if ((W * H & 0x1) == 1) {
          W = align_up(W * H, (int64_t)2) / 2;
          real_wslice = align_up(real_wslice * real_hslice, (int64_t)2) / 2;
          H = 1;
          real_hslice = 1;
        } else {
          if ((W & 0x1) == 1) {
            real_hslice >>= 1;
            H >>= 1;
          } else {
            W >>= 1;
            real_wslice >>= 1;
          }
        }
      } else {
        real_hslice = gi.h_slice; // to do for int4
      }
    }
  } else {
    module::getNCDHW(getOutput(), N, C, D, H, W, group_type);
  }

  gdma_format = BM168x::getGdmaFormat(data_type);
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  auto g_addr = module::getAddress(getOutput());
  int64_t c_num_local = ceiling_func(real_cslice, Arch::NPU_NUM);
  int64_t c_stride =
      gi.eu_align ? align_up(real_hslice * real_wslice, Arch::eu_num(fmt_bytes))
                  : real_hslice * real_wslice;
  int64_t channel_num = real_cslice;
  if (real_dslice <= gi.n_slice) {
    const int64_t csecs = ceiling_func(channel_num, (int64_t)MAX_TPU_DIM);
    for (int64_t d = 0; d < real_dslice; d++) {
      int64_t channel_index = 0;
      while (channel_index < csecs) {
        int64_t cur_cslice =
            std::min(channel_num - channel_index * (int64_t)MAX_TPU_DIM,
                     (int64_t)MAX_TPU_DIM);
        int64_t real_c_num_local =
            (channel_index * (int64_t)MAX_TPU_DIM) / Arch::NPU_NUM;
        int64_t src_offset_c = real_c_num_local * c_stride * fmt_bytes;
        int64_t dst_offset_c =
            (channel_index * (int64_t)MAX_TPU_DIM + gi.c_idx) * H * W *
            fmt_bytes;
        int64_t real_npu_idx =
            (channel_index * (int64_t)MAX_TPU_DIM) % Arch::NPU_NUM;
        int64_t cur_local_offset =
            d * gi.n_slice * c_num_local * c_stride * fmt_bytes + src_offset_c;
        int64_t cur_global_offset = gi.n_idx * C * D * H * W * fmt_bytes +
                                    (gi.d_idx + d) * H * W * fmt_bytes +
                                    gi.h_idx * W * fmt_bytes +
                                    gi.w_idx * fmt_bytes + dst_offset_c;
        BM168x::instance()->dl_tensor_stride_move_gen_cmd(
            gi.out_addr + cur_local_offset, real_npu_idx,
            g_addr + cur_global_offset, gi.n_slice, cur_cslice, real_hslice,
            real_wslice, c_num_local * c_stride, c_stride, real_wslice, 1,
            C * D * H * W, D * H * W, W, 1, gdma_format,
            GDMA_VALUE_DIR_L2S, // 1,
            0, pid_node);
        channel_index++;
      }
    }
  } else { // HAVE DEPTH,3D [D,n_slice,C,h_slice,W] -> [N,C,D,H,W]
    for (int64_t i = 0; i < gi.n_slice; i++) {
      int64_t cur_local_offset = i * c_num_local * c_stride * fmt_bytes;
      int64_t cur_global_offset =
          (gi.n_idx + i) * C * D * H * W * fmt_bytes +
          gi.c_idx * D * H * W * fmt_bytes + gi.d_idx * H * W * fmt_bytes +
          gi.h_idx * W * fmt_bytes + gi.w_idx * fmt_bytes;
      BM168x::instance()->dl_tensor_stride_move_gen_cmd(
          gi.out_addr + cur_local_offset, 0, g_addr + cur_global_offset,
          real_dslice, real_cslice, real_hslice, real_wslice,
          gi.n_slice * c_num_local * c_stride, c_stride, real_wslice, 1, H * W,
          D * H * W, W, 1, gdma_format,
          GDMA_VALUE_DIR_L2S, // 1,
          0, pid_node);
    }
  }
}

// dynamic codegen
int64_t tpu::StoreOp::dyn_codegen_local_bm1684x(void *buffer) { return 0; }

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::StoreOp::dyn_codegen_global_bm1684x(void *buffer) { return 0; }

int64_t tpu::StoreOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }

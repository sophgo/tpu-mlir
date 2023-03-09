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
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DynCompileCommon.hpp"
using namespace tpu_mlir::backend;

void tpu::StoreOp::codegen_global_bm1684x() {
  llvm_unreachable("Not Implemented");
}

int64_t tpu::StoreOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice, group_type_t group_type) {
  return 0;
}

void tpu::StoreOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                         group_type_t group_type,
                                         local_sec_info_t &sec_info) {
  CMD_ID_NODE *pid_node = (CMD_ID_NODE *)BM168x::instance()->gdma_node;
  auto gi = getGroupInfo(n_step, h_step);
  int64_t N, C, H, W, real_hslice;
  int gdma_format;
  module::getNCHW(getOutput(), N, C, H, W, group_type);
  auto data_type = BM168x::getDataType(getOutput());

  real_hslice = gi.h_slice;
  if (data_type == DTYPE_UINT4 || data_type == DTYPE_INT4) {
    gdma_format = BM168x::GDMA_VALUE_FORMAT_INT8;
    data_type = DTYPE_INT8;
    if (gi.h_slice == H) {
      if (W * H & 0x1 == 1) {
        W = align_up(W * H, (int64_t)2) / 2;
        H = 1;
        real_hslice = 1;
      } else {
        if (W & 0x1 == 1)
          real_hslice >>= 1;
        else
          W >>= 1;
      }
    } else {
      real_hslice = gi.h_slice;   // to do for int4
    }
  }
  gdma_format = BM168x::getGdmaFormat(data_type);
  auto fmt_bytes = BM168x::getFmtBytes(data_type);

  auto g_stride = BM168x::getGlobalStride(N, C, H, W);
  auto s_stride = BM168x::getLocalStride(
      gi.n_slice, C, real_hslice, W, fmt_bytes, gi.eu_align);
  auto g_addr = module::getAddress(getOutput());
  int64_t g_offset =
      (gi.n_idx * g_stride.N + gi.h_idx * g_stride.H) * fmt_bytes;
  BM168x::instance()->dl_tensor_stride_move_gen_cmd(
      gi.out_addr, 0, g_addr + g_offset, gi.n_slice, C,
      real_hslice, W, s_stride.N, s_stride.C, s_stride.H,
      s_stride.W, g_stride.N, g_stride.C, g_stride.H, g_stride.W, gdma_format,
      GDMA_VALUE_DIR_L2S, 0, pid_node);
}

// dynamic codegen
int64_t tpu::StoreOp::dyn_codegen_local_bm1684x(void *buffer) { return 0; }

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::StoreOp::dyn_codegen_global_bm1684x(void *buffer) { return 0; }

int64_t tpu::StoreOp::get_fw_type_bm1684x() {
  return FW_LAYER_UNKNOWN;
}

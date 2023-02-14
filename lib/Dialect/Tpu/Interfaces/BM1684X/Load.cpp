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

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::LoadOp::codegen_global_bm1684x() {
  llvm_unreachable("global not support");
}

int64_t tpu::LoadOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                           int64_t out_lmem_bytes,
                                           int64_t in_nslice, int64_t in_hslice,
                                           int64_t out_nslice,
                                           int64_t out_hslice) {
  return 0;
}

void tpu::LoadOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                        local_sec_info_t &sec_info) {
  auto pid_node = (CMD_ID_NODE *)BM168x::instance()->gdma_node;
  auto gi = getGroupInfo(n_step, h_step);
  assert(false == gi.overstepped);

  int64_t N, C, H, W, real_hslice;
  int gdma_format;
  module::getNCHW(getOutput(), N, C, H, W);
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
      real_hslice = gi.h_slice;     // to do for int4
    }
  }
  gdma_format = BM168x::getGdmaFormat(data_type);
  auto fmt_bytes = BM168x::getFmtBytes(data_type);

  auto g_stride = BM168x::getGlobalStride(N, C, H, W);

  if (getDoBcast() == true) {
    C = BM168x::NPU_NUM;
    g_stride.N = 0;
    g_stride.C = 0;
    g_stride.H = 0;
  }
  auto s_stride = BM168x::getLocalStride(gi.n_slice, C, real_hslice, W,
                                         fmt_bytes, gi.eu_align);
  auto g_addr = module::getAddress(getInput());
  int64_t g_offset =
      (gi.n_idx * g_stride.N + gi.h_idx * g_stride.H) * fmt_bytes;
  int64_t use_3ic = getUse_3icOptimize();
  if (use_3ic < 4 && use_3ic > 0) {
    auto use_op = *getOutput().getUsers().begin();
    auto conv_op = dyn_cast<tpu::Conv2DOp>(use_op);
    auto kernel = module::getI64Array(conv_op.getKernelShape());
    int64_t to_ic =
        use_3ic == 1
            ? kernel->at(0)
            : (use_3ic == 2 ? kernel->at(1) : kernel->at(0) * kernel->at(1));
    for (int i = 0; i < C; ++i) {
      BM168x::instance()->dl_tensor_broadcast_move_gen_cmd(
          g_addr + g_offset + i * W * H * fmt_bytes, 0, gi.out_addr, i * to_ic,
          gi.n_slice, real_hslice, W, to_ic, g_stride.N, g_stride.H, s_stride.N,
          s_stride.H, gdma_format, true, GDMA_VALUE_DIR_S2L, pid_node);
    }
  } else {
    BM168x::instance()->dl_tensor_stride_move_gen_cmd(
        gi.out_addr, 0, g_addr + g_offset, gi.n_slice, C, real_hslice, W, g_stride.N,
        g_stride.C, g_stride.H, g_stride.W, s_stride.N, s_stride.C, s_stride.H,
        s_stride.W, gdma_format, GDMA_VALUE_DIR_S2L, 0, pid_node);
  }
}

// dynamic codegen
int64_t tpu::LoadOp::dyn_codegen_local_bm1684x(void *buffer) {
  // no need to implement it
  return 0;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::LoadOp::dyn_codegen_global_bm1684x(void *buffer) {
  // no need to implement it
  return 0;
}

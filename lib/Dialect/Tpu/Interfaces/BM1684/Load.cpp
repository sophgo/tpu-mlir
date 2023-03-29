//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"

#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::LoadOp::codegen_global_bm1684() {
  llvm_unreachable("Not Implemented");
}

int64_t tpu::LoadOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  return 0;
}

void tpu::LoadOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                       local_sec_info_t &sec_info) {
  auto pid_node = (CMD_ID_NODE *)BM168x::instance()->gdma_node;
  auto gi = getGroupInfo(n_step, h_step, 0, 0);
  assert(false == gi.overstepped);
  auto data_type = BM168x::getDataType(getOutput());
  auto gdma_format = BM168x::getGdmaFormat(data_type);
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  int64_t N, C, H, W;
  module::getNCHW(getOutput(), N, C, H, W);
  if (getDoBcast()) {
    C = BM168x::NPU_NUM;
  }
  int64_t n_idx = gi.n_idx;
  int64_t local_N = gi.n_slice, local_C = C, local_H = gi.h_slice, local_W = W;
  if (fmt_bytes != 4) {
    if (!module::isWeight(getInput())) {
      int64_t N_align = 4 / fmt_bytes;
      fmt_bytes = 4;
      gdma_format = BM168x::GDMA_VALUE_FORMAT_FLOAT32;
      N = ceiling_func(N, N_align);
      local_N = ceiling_func(gi.n_slice, N_align);
      n_idx = gi.n_idx / 4;
      assert(gi.n_idx % 4 == 0);
    }
  }
  auto g_stride = BM168x::getGlobalStride(N, C, H, W);
  if (getDoBcast() == true) {
    g_stride.N = 0;
    g_stride.C = 0;
    g_stride.H = 0;
  }
  auto s_stride = BM168x::getLocalStride(local_N, local_C, local_H, local_W,
                                         fmt_bytes, gi.eu_align);
  auto g_addr = module::getAddress(getInput());
  int64_t g_offset = (n_idx * g_stride.N + gi.h_idx * g_stride.H) * fmt_bytes;

  BM168x::instance()->dl_tensor_stride_move_gen_cmd(
      gi.out_addr, 0, g_addr + g_offset, local_N, local_C, local_H, local_W,
      g_stride.N, g_stride.C, g_stride.H, g_stride.W, s_stride.N, s_stride.C,
      s_stride.H, s_stride.W, gdma_format, GDMA_VALUE_DIR_S2L, 0, pid_node);
}

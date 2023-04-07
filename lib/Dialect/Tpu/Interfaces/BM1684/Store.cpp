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

void tpu::StoreOp::codegen_global_bm1684() {
  llvm_unreachable("Not Implemented");
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
  CMD_ID_NODE *pid_node = (CMD_ID_NODE *)BM168x::instance()->gdma_node;
  auto gi = getGroupInfo(n_step, h_step, 0, 0);
  auto data_type = BM168x::getDataType(getOutput());
  auto gdma_format = BM168x::getGdmaFormat(data_type);
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  int64_t N, C, H, W;
  module::getNCHW(getOutput(), N, C, H, W);
  int64_t n_idx = gi.n_idx;
  int64_t local_N = gi.n_slice, local_C = C, local_H = gi.h_slice, local_W = W;
  if (fmt_bytes != 4) {
    int64_t N_align = 4 / fmt_bytes;
    fmt_bytes = 4;
    gdma_format = BM168x::GDMA_VALUE_FORMAT_FLOAT32;
    N = ceiling_func(N, N_align);
    local_N = ceiling_func(gi.n_slice, N_align);
    n_idx = gi.n_idx / 4;
    assert(gi.n_idx % 4 == 0);
  }
  auto g_stride = BM168x::getGlobalStride(N, C, H, W);
  auto s_stride = BM168x::getLocalStride(local_N, local_C, local_H, local_W,
                                         fmt_bytes, gi.eu_align);
  auto g_addr = module::getAddress(getOutput());
  int64_t g_offset = (n_idx * g_stride.N + gi.h_idx * g_stride.H) * fmt_bytes;
  BM168x::instance()->dl_tensor_stride_move_gen_cmd(
      gi.out_addr, 0, g_addr + g_offset, local_N, local_C, local_H, local_W,
      s_stride.N, s_stride.C, s_stride.H, s_stride.W, g_stride.N, g_stride.C,
      g_stride.H, g_stride.W, gdma_format, GDMA_VALUE_DIR_L2S, 0, pid_node);
}

uint32_t tpu::StoreOp::dyn_codegen_global_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}
int64_t tpu::StoreOp::get_fw_type_bm1684() {
  return -1;
}

int32_t tpu::StoreOp::dyn_codegen_local_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}
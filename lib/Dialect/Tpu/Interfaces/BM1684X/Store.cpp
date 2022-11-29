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
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

void tpu::StoreOp::codegen_global_bm1684x() {
  llvm_unreachable("Not Implemented");
}

int64_t tpu::StoreOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::StoreOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  CMD_ID_NODE *pid_node = (CMD_ID_NODE *)BM168x::instance()->gdma_node;
  auto gi = getGroupInfo(n_step, h_step);
  auto data_type = BM168x::getDataType(output());
  auto gdma_format = BM168x::getGdmaFormat(data_type);
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  int64_t N, C, H, W;
  Module::getNCHW(output(), N, C, H, W);
  auto g_stride = BM168x::getGlobalStride(N, C, H, W);
  auto s_stride = BM168x::getLocalStride(gi.n_slice, C, gi.h_slice, W,
                                         fmt_bytes, gi.eu_align);
  auto g_addr = Module::getAddress(output());
  int64_t g_offset =
      (gi.n_idx * g_stride.N + gi.h_idx * g_stride.H) * fmt_bytes;
  BM168x::instance()->dl_tensor_stride_move_gen_cmd(
      gi.out_addr, 0, g_addr + g_offset, gi.n_slice, C, gi.h_slice, W,
      s_stride.N, s_stride.C, s_stride.H, s_stride.W, g_stride.N, g_stride.C,
      g_stride.H, g_stride.W, gdma_format, GDMA_VALUE_DIR_L2S, 0, pid_node);
}

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

void tpu::LoadOp::assign_sec_info(int64_t n_step, int64_t h_step,
                                   void *sec_info_) {
  local_sec_info_t *sec_info = (local_sec_info_t *)sec_info_;
  memset(sec_info, 0, sizeof(local_sec_info_t));

  int64_t n, c, h, w;
  module::getNCHW(input(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  sec_info->n_slice = gi.n_slice;
  sec_info->d_slice = 1;
  sec_info->h_slice = gi.h_slice;
  sec_info->h_idx = gi.h_idx;
  sec_info->is_h_split = !(gi.h_idx == 0 && gi.h_slice == h);
  sec_info->w_slice = w;
  sec_info->out_n_slice = gi.n_slice;
  sec_info->out_h_idx = gi.h_idx;
  sec_info->out_h_slice = gi.h_slice;
  sec_info->out_w_slice = w;
}

void tpu::LoadOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                        void *sec_info_) {
  auto pid_node = (CMD_ID_NODE *)BM168x::instance()->gdma_node;
  auto gi = getGroupInfo(n_step, h_step);
  assert(false == gi.overstepped);
  auto data_type = BM168x::getDataType(output());
  auto gdma_format = BM168x::getGdmaFormat(data_type);
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  int64_t N, C, H, W;
  module::getNCHW(output(), N, C, H, W);
  auto g_stride = BM168x::getGlobalStride(N, C, H, W);

  if (do_bcast() == true) {
    C = BM168x::NPU_NUM;
    g_stride.N = 0;
    g_stride.C = 0;
    g_stride.H = 0;
  }
  auto s_stride = BM168x::getLocalStride(gi.n_slice, C, gi.h_slice, W,
                                         fmt_bytes, gi.eu_align);
  auto g_addr = module::getAddress(input());
  int64_t g_offset =
      (gi.n_idx * g_stride.N + gi.h_idx * g_stride.H) * fmt_bytes;
  int64_t use_3ic = use_3ic_optimize();
  if (use_3ic < 4 && use_3ic > 0) {
    auto use_op = *output().getUsers().begin();
    auto conv_op = dyn_cast<tpu::Conv2DOp>(use_op);
    auto kernel = module::getI64Array(conv_op.kernel_shape());
    int64_t to_ic =
        use_3ic == 1
            ? kernel->at(0)
            : (use_3ic == 2 ? kernel->at(1) : kernel->at(0) * kernel->at(1));
    for (int i = 0; i < C; ++i) {
      BM168x::instance()->dl_tensor_broadcast_move_gen_cmd(
          g_addr + g_offset + i * W * H * fmt_bytes, 0, gi.out_addr, i * to_ic,
          gi.n_slice, gi.h_slice, W, to_ic, g_stride.N, g_stride.H, s_stride.N,
          s_stride.H, gdma_format, true, GDMA_VALUE_DIR_S2L, pid_node);
    }
  } else {
    BM168x::instance()->dl_tensor_stride_move_gen_cmd(
        gi.out_addr, 0, g_addr + g_offset, gi.n_slice, C, gi.h_slice, W,
        g_stride.N, g_stride.C, g_stride.H, g_stride.W, s_stride.N, s_stride.C,
        s_stride.H, s_stride.W, gdma_format, GDMA_VALUE_DIR_S2L, 0, pid_node);
  }
}

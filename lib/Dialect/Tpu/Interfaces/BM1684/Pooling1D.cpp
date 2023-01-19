//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Pool.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

void tpu::Pool1DOp::codegen_global_bm1684() {
  auto attr = parseParam();
  bool is_avg_pooling = getPoolMode() == tpu::PoolMode::Avg;
  BM1684::instance().dl_nodechip_pooling_fix8b_forward_parallel_with_data_split(
      module::getAddress(getInput()), module::getAddress(getOutput()), attr.n,
      attr.c, attr.ih, attr.iw, attr.kh, attr.kw, attr.pad_h, attr.pad_h_after,
      attr.pad_w, attr.pad_w_after, attr.sh, attr.sw, 0, 0, is_avg_pooling, 0,
      0, 0, 0, 1, 0, 0, 1, attr.do_relu ? 1 : 0,
      (CMD_ID_NODE *)BM1684::instance().cmdid_node);
}

int64_t tpu::Pool1DOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::Pool1DOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  llvm_unreachable("Not Implemented");
}

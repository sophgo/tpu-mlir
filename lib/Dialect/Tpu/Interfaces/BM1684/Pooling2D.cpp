//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Dnnl/Pool.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

void tpu::Pool2DOp::codegen_global_bm1684() {
  pool_attr_t attrs;
  parseParam(&attrs);
  bool is_avg_pooling = pool_mode() == tpu::PoolMode::Avg;
  BM1684::instance().dl_nodechip_pooling_fix8b_forward_parallel_with_data_split(
      Module::getAddress(input()), Module::getAddress(output()), attrs.n,
      attrs.c, attrs.ih, attrs.iw, attrs.kh, attrs.kw, attrs.pad_h,
      attrs.pad_h_after, attrs.pad_w, attrs.pad_w_after, attrs.sh, attrs.sw, 0,
      0, is_avg_pooling, 0, 0, 0, 0, 1, 0, 0, 1, attrs.do_relu ? 1 : 0,
      (CMD_ID_NODE *)BM1684::instance().cmdid_node);
}

int64_t tpu::Pool2DOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::Pool2DOp::codegen_local_bm1684(int64_t n_step, int64_t h_step) {
  llvm_unreachable("support later");
}

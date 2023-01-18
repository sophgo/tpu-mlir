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

#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/Dnnl/Pool.h"

using namespace tpu_mlir::backend;

void tpu::Pool2DOp::codegen_global_bm1684() {
  auto attr = parseParam();
  bool is_avg_pooling = getPoolMode() == tpu::PoolMode::Avg;
  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  if (module::isUniformQuantized(getInput())) {
    auto in_sign = module::isSign(getInput());
    auto out_sign = module::isSign(getOutput());
    BM1684::instance()
        .dl_nodechip_pooling_fix8b_forward_parallel_with_data_split(
            in_addr, out_addr, attr.n, attr.c, attr.ih, attr.iw, attr.kh,
            attr.kw, attr.pad_h, attr.pad_h_after, attr.pad_w, attr.pad_w_after,
            attr.sh, attr.sw, 0, 0, is_avg_pooling ? 1 : 0, 0, 0, 0, 0, in_sign,
            0, 0, out_sign, attr.do_relu ? 1 : 0,
            (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  } else {
    // F32
    BM1684::instance().dl_nodechip_pooling_forward_parallel_with_data_split(
        in_addr, out_addr, attr.n, attr.c, attr.ih, attr.iw, attr.oh, attr.ow,
        attr.kh, attr.kw, attr.pad_h, attr.pad_w, attr.sh, attr.sw,
        is_avg_pooling ? 1 : 0, attr.count_include_pad ? 0 : 1,
        attr.do_relu ? 1 : 0, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  }
}

int64_t tpu::Pool2DOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::Pool2DOp::codegen_local_bm1684(int64_t n_step, int64_t h_step) {
  llvm_unreachable("Not Implemented");
}

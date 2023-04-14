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

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

void tpu::ReshapeOp::codegen_global_bm1684() {
  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  if (in_addr == out_addr) {
    return;
  }
  int64_t in, ic, ih, iw, on, oc, oh, ow;
  module::getNCHW(getInput(), in, ic, ih, iw);
  module::getNCHW(getOutput(), on, oc, oh, ow);
  if (on != in) {
    llvm_unreachable("Not Implemented");
  } else {
    int total_num = align_up(on, 4ll) * oc * oh * ow;
    BM1684::instance().dl_nodechip_global_memcpy_ex(
        in_addr, out_addr, 1, total_num, total_num, DTYPE_FP32, DTYPE_FP32,
        total_num, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  }
}

// ======================================
// LocalGenInterface
// ======================================
int64_t tpu::ReshapeOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::ReshapeOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  // do nothing
  llvm_unreachable("Not supported now");
}

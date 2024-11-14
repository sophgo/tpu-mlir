//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::LRNOp::codegen_global_bm1684() {

  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  if (false == module::isUniformQuantized(getInput())) {
    BM1684::instance().dl_nodechip_lrn_forward_parallel(
        in_addr, out_addr, n, c, h, w, getAlpha().convertToDouble(), getSize(),
        getBeta().convertToDouble(), getBias().convertToDouble(),
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    int in_sign = module::isSign(getInput());
    BM1684::instance().dl_nodechip_lrn_fix8b_forward_parallel(
        in_addr, out_addr, n, c, h, w, in_sign, getAlpha().convertToDouble(),
        getSize(), getBeta().convertToDouble(), getBias().convertToDouble(), 1,
        1, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

int64_t tpu::LRNOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes,
                                         int64_t in_nslice, int64_t in_hslice,
                                         int64_t out_nslice,
                                         int64_t out_hslice) {
  llvm_unreachable("not supported now");
  return 0;
}

void tpu::LRNOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                      local_sec_info_t &sec_info) {
  UNREACHABLE_THIS("Not Implemented");
}

uint32_t tpu::LRNOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::LRNOp::get_fw_type_bm1684() { return -1; }

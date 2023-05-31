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
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::Pool3DOp::codegen_global_bm1684() {
  auto attr = parseParam();
  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  auto buffer_addr = module::getAddress(getBuffer());

  int is_avg_pooling = getPoolMode() == tpu::PoolMode::Avg ? 1 : 0;
  int avg_pooling_mode = attr.count_include_pad ? 0 : 1;
  if (module::isUniformQuantized(getInput())) {
    auto in_sign = module::isSign(getInput());
    auto out_sign = module::isSign(getOutput());
    if (is_avg_pooling == 0) {
      BM1684::instance().dl_nodechip_pooling3d_fix8b_forward_parallel(
          in_addr, buffer_addr, out_addr, attr.n, attr.c, attr.id, attr.ih,
          attr.iw, attr.kd, attr.kh, attr.kw, attr.pad_d, attr.pad_d_after,
          attr.pad_h, attr.pad_h_after, attr.pad_w, attr.pad_w_after, attr.sd,
          attr.sh, attr.sw, is_avg_pooling, avg_pooling_mode, 0, 0, 0, in_sign,
          0, 0, out_sign, attr.do_relu ? 1 : 0, attr.relu_limit,
          (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    } else {
      llvm_unreachable("nodechip currently not supported.");
    }
  } else {
    BM1684::instance().dl_nodechip_pooling3d_forward_parallel(
        in_addr, buffer_addr, out_addr, attr.n, attr.c, attr.id, attr.ih,
        attr.iw, attr.od, attr.oh, attr.ow, attr.kd, attr.kh, attr.kw,
        attr.pad_d, attr.pad_d_after, attr.pad_h, attr.pad_h_after, attr.pad_w,
        attr.pad_w_after, attr.sd, attr.sh, attr.sw, is_avg_pooling,
        avg_pooling_mode, attr.do_relu ? 1 : 0, attr.relu_limit,
        (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  }
}

int64_t tpu::Pool3DOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::Pool3DOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  llvm_unreachable("Not Implemented");
}

uint32_t tpu::Pool3DOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}
int64_t tpu::Pool3DOp::get_fw_type_bm1684() { return -1; }

int32_t tpu::Pool3DOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}

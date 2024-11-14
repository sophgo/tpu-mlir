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

using namespace tpu_mlir::backend;

void tpu::UpsampleOp::codegen_global_bm1684() {
  int64_t n, c, h, w;
  assert(getScaleH() == getScaleW());
  module::getNCHW(getInput(), n, c, h, w);
  if (module::isUniformQuantized(getInput())) {
    BM1684::instance().dl_nodechip_upsample_forward_parallel_fix8b(
        module::getAddress(getInput()), module::getAddress(getOutput()), n, c,
        h, w, getScaleH(), getDoRelu() ? 1 : 0,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    BM1684::instance().dl_nodechip_upsample_forward_parallel_with_data_split(
        module::getAddress(getInput()), module::getAddress(getOutput()), n, c,
        h, w, getScaleH(), getDoRelu() ? 1 : 0,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

int64_t tpu::UpsampleOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::UpsampleOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                           local_sec_info_t &sec_info) {
  auto out_gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  int scale = getScaleH();
  assert(scale == getScaleW());
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  int bottom_dim[4] = {(int)in_gi.n_slice, (int)c, (int)in_gi.h_slice, (int)w};
  int top_dim[4] = {(int)out_gi.n_slice, (int)c, (int)out_gi.h_slice,
                    (int)w * scale};
  uint32_t bottom_local_offset = in_gi.out_addr;
  uint32_t top_local_offset = out_gi.out_addr;
  if (module::isUniformQuantized(getInput())) {
    BM1684::instance().dl_nodechip_upsample_fix8b_forward_local(
        bottom_local_offset, top_local_offset, bottom_dim, top_dim, scale, 0, 1,
        1, 1, 1, (CMD_ID_NODE *)BM1684::instance()->bdc_node,
        getDoRelu() ? 1 : 0);
  } else {
    BM1684::instance().dl_nodechip_upsample_forward_local(
        bottom_local_offset, top_local_offset, bottom_dim, top_dim, scale,
        (CMD_ID_NODE *)BM1684::instance()->bdc_node, getDoRelu() ? 1 : 0);
  }
}

uint32_t tpu::UpsampleOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::UpsampleOp::get_fw_type_bm1684() { return -1; }

int32_t tpu::UpsampleOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

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

void tpu::ShuffleChannelOp::codegen_global_bm1684() {
  auto input_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  if (module::isUniformQuantized(getInput())) {
    BM1684::instance().dl_nodechip_shuffle_channel_fix8b_forward(
        input_addr, out_addr, n, c, h, w, getGroup(),
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    BM1684::instance().dl_nodechip_shuffle_channel_forward(
        input_addr, out_addr, n, c, h, w, getGroup(),
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

uint32_t tpu::ShuffleChannelOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::ShuffleChannelOp::get_fw_type_bm1684() { return -1; }

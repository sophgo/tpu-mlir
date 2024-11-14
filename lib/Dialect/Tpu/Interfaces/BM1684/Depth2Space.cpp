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

void tpu::Depth2SpaceOp::codegen_global_bm1684() {
  auto input = getInput();
  auto output = getOutput();
  auto input_addr = module::getAddress(input);
  auto output_addr = module::getAddress(output);
  int input_dims = module::getShape(input).size();
  int input_shape[MAX_SHAPE_DIMS] = {0};
  int64_t in, ic, ih, iw;
  module::getNCHW(getInput(), in, ic, ih, iw, false);
  input_shape[0] = (int32_t)in;
  input_shape[1] = (int32_t)ic;
  input_shape[2] = (int32_t)ih;
  input_shape[3] = (int32_t)iw;
  int block_sizes[2] = {(int)getBlockH(), (int)getBlockW()};

  BM1684::instance().dl_nodechip_depth2space_mlir(
      input_addr, output_addr, input_shape, input_dims, block_sizes,
      getInIs_NCHW(), getOutIs_NCHW(), getIsInversed(), getIs_CRD(),
      getSwapCr(), module::getDtypeSize(input),
      (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
uint32_t tpu::Depth2SpaceOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::Depth2SpaceOp::get_fw_type_bm1684() { return -1; }

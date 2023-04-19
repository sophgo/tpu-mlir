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

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::MulShiftOp::codegen_global_bm1684() {
  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  int in_shape[MAX_SHAPE_DIMS];
  module::getGlobalShape(getInput(), in_shape);
  int sign[3];
  sign[0] = module::isSign(getInput());
  sign[1] = 0;
  sign[2] = module::isSign(getOutput());
  BM1684::instance().dl_nodechip_mulshift_fix8b_forward(
      in_addr, out_addr, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
      getMultiplier(), getRshift(), sign[0], sign[1], sign[2],
      (CMD_ID_NODE *)BM1684::instance().cmdid_node);
}

int64_t tpu::MulShiftOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  if (module::isSign(getInput()) == 1 && module::isSign(getOutput()) == 0) {
    int64_t buffer_size =
        ceiling_func(out_nslice, (int64_t)2) *
        ceiling_func(c, BM1684::NPU_NUM) *
        align_up(out_hslice * w, BM1684::eu_num(sizeof(int))) * sizeof(int);
    return buffer_size;
  } else {
    return 0;
  }
}

void tpu::MulShiftOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                           local_sec_info_t &sec_info) {
  auto in_g_info = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step, 0, 0);
  // int in_shape[MAX_SHAPE_DIMS];
  // module::getLocalShape(getInput(), n_step, h_step, in_shape);
  int out_shape[MAX_SHAPE_DIMS];
  module::getLocalShape(getOutput(), n_step, h_step, out_shape);
  int sign[3];
  sign[0] = module::isSign(getInput());
  sign[1] = 1;
  sign[2] = module::isSign(getOutput());
  BM1684::instance().dl_nodechip_mulshift_fix8b_forward_local(
      in_g_info.out_addr, gi.buffer_addr, gi.out_addr, out_shape[0],
      out_shape[1], out_shape[2], out_shape[3], getMultiplier(), getRshift(),
      sign[0], sign[1], sign[2], (CMD_ID_NODE *)BM1684::instance().bdc_node);
}

uint32_t tpu::MulShiftOp::dyn_codegen_global_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}
int64_t tpu::MulShiftOp::get_fw_type_bm1684() {
  return -1;
}

int32_t tpu::MulShiftOp::dyn_codegen_local_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}
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

void tpu::SwapDimInnerOp::codegen_global_bm1684() {
  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  auto in_dims = module::getShape(getInput()).size();
  assert(in_dims <= 4);
  auto offset = module::getI64Array(getOffset());
  int in_shape[MAX_SHAPE_DIMS];
  int axis_list[MAX_SHAPE_DIMS];
  int offset_list[MAX_SHAPE_DIMS];
  module::getGlobalShape(getInput(), in_shape, in_dims);
  int axis_num = 0;
  for (int i = 0; i < offset->size(); ++i) {
    if (offset->at(i) != 0) {
      axis_list[axis_num] = i;
      offset_list[axis_num] = offset->at(i);
      axis_num += 1;
    }
  }

  if (module::isUniformQuantized(getInput())) {
    in_shape[0] = (in_shape[0] + 3) >> 2;
    BM1684::instance().dl_nodechip_swap_dim(
        in_addr, out_addr, in_shape, in_dims, axis_num, axis_list, offset_list,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else { // FP32
    BM1684::instance().dl_nodechip_swap_dim(
        in_addr, out_addr, in_shape, in_dims, axis_num, axis_list, offset_list,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::SwapDimInnerOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::SwapDimInnerOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                               local_sec_info_t &sec_info) {
  UNREACHABLE_THIS("Not Implemented");
}

uint32_t tpu::SwapDimInnerOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::SwapDimInnerOp::get_fw_type_bm1684() { return -1; }

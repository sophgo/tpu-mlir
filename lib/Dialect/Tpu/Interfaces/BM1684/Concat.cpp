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

void tpu::ConcatOp::codegen_global_bm1684() {
  if (getOnlyMerge()) {
    return;
  }
  int num_input = getInputs().size();
  int(*bottomtensor_shape)[MAX_SHAPE_DIMS] = new int[num_input][MAX_SHAPE_DIMS];
  int is_st_concat_way[num_input];
  memset(is_st_concat_way, 0, sizeof(int) * (num_input + 1));
  uint64_t in_addr[num_input];
  memset(in_addr, 0, sizeof(int) * (num_input + 1));
  auto out_addr = module::getAddress(getOutput());
  for (int i = 0; i < num_input; ++i) {
    in_addr[i] = module::getAddress(getInputs()[i]);
    module::getGlobalShape(getInputs()[i], bottomtensor_shape[i]);
  }
  int out_shape[MAX_SHAPE_DIMS] = {0};
  module::getGlobalShape(getOutput(), out_shape);
  BM1684::instance().dl_nodechip_concat_md(
      getAxis(), module::getShape(getInputs()[0]).size(), getInputs().size(),
      in_addr, out_addr, bottomtensor_shape, out_shape, is_st_concat_way,
      (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  delete[] bottomtensor_shape;
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ConcatOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::ConcatOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  int num_inputs = getInputs().size();
  int is_st_concat_way[num_inputs];
  memset(is_st_concat_way, 0, sizeof(int) * (num_inputs + 1));
  uint32_t in_addr[num_inputs];
  memset(in_addr, 0, sizeof(int) * (num_inputs + 1));
  auto bottomtensor_shape = new int *[num_inputs];
  for (int i = 0; i < num_inputs; i++) {
    in_addr[i] = LocalGenInterface::getGroupInfo(getInputs()[i], n_step, h_step)
                     .out_addr;
    bottomtensor_shape[i] = new int[4];
    module::getLocalShape(getInputs()[i], n_step, h_step,
                          bottomtensor_shape[i]);
  }
  int out_shape[MAX_SHAPE_DIMS] = {0};
  module::getLocalShape(getOutput(), n_step, h_step, out_shape);
  BM1684::instance().dl_nodechip_concat_local_v2(
      in_addr, gi.out_addr, bottomtensor_shape,
      module::getShape(getInputs()[0]).size(), is_st_concat_way, out_shape,
      getAxis(), (CMD_ID_NODE *)BM1684::instance().bdc_node,
      (CMD_ID_NODE *)BM1684::instance().gdma_node);
  for (int i = 0; i < num_inputs; ++i) {
    delete[] bottomtensor_shape[i];
  }
  delete[] bottomtensor_shape;
}

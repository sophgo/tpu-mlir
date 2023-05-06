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

void tpu::LutOp::codegen_global_bm1684() {
  auto data_type = BM168x::getDataType(getInput());
  assert(data_type == DTYPE_INT8 || data_type == DTYPE_UINT8);
  auto dtype_o = BM168x::getDataType(getOutput());

  int out_dtype = dtype_o == DTYPE_INT8 || dtype_o == DTYPE_UINT8 ? 1 : 0;
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  BM1684::instance().dl_nodechip_lut_v2(
      module::getAddress(getInput()), module::getAddress(getOutput()),
      module::getAddress(getTable()), n, c, h, w,
      /*in store_mode*/ 0, out_dtype,
      /*out store_mode*/ 1, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
}

int64_t tpu::LutOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes,
                                         int64_t in_nslice, int64_t in_hslice,
                                         int64_t out_nslice,
                                         int64_t out_hslice) {
  uint64_t buffer_size = 0;
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  int shape[4] = {(int)in_nslice, (int)c, (int)in_hslice, (int)w};
  shape[0] = align_up(shape[0], 4);
  int cnum = (shape[1] + BM1684::NPU_NUM - 1) / BM1684::NPU_NUM;
  int cstride =
      align_up(shape[2] * shape[3], (int)BM1684::eu_num(sizeof(std::int8_t)) * 4);
  buffer_size = n * cnum * cstride;
  return buffer_size * 2;
}

void tpu::LutOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                      local_sec_info_t &sec_info) {
  auto out_g_info = getGroupInfo(n_step, h_step, 0, 0);
  auto in_g_info = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto tb_g_info = LocalGenInterface::getGroupInfo(getTable(), n_step, h_step);
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  int b0_shape[MAX_SHAPE_DIMS] = {(int)in_g_info.n_slice, (int)c,
                                  (int)in_g_info.h_slice, (int)w};
  BM1684::instance().dl_nodechip_lut_local_v2(
      in_g_info.out_addr, tb_g_info.out_addr, out_g_info.buffer_addr,
      out_g_info.out_addr, b0_shape, module::getShape(getOutput()).size(), 1,
      DTYPE_INT8, 0, (CMD_ID_NODE *)BM1684::instance().bdc_node);
}

uint32_t tpu::LutOp::dyn_codegen_global_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}
int64_t tpu::LutOp::get_fw_type_bm1684() {
  return -1;
}

int32_t tpu::LutOp::dyn_codegen_local_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}

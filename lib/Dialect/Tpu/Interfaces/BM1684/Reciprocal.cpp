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

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::ReciprocalOp::codegen_global_bm1684() {
  assert(!module::isUniformQuantized(getInput()));
  uint64_t bottom_global_offset = module::getAddress(getInput());
  uint64_t top_global_offset = module::getAddress(getOutput());
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  uint64_t length = n * c * h * w;
  auto B_const_val = getConstVal().convertToDouble();
  int binary_type = BINARY_DIV;
  int inversed = 1;
  int if_relu = getDoRelu();
  float relu_limit = getReluLimit().convertToDouble();
  BM1684::instance().dl_nodechip_const_binary(
      bottom_global_offset, length, B_const_val, top_global_offset, binary_type,
      inversed, if_relu, relu_limit,
      (CMD_ID_NODE *)BM1684::instance()->cmdid_node,
      module::getStorageType(getInput()).isa<IntegerType>());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ReciprocalOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::ReciprocalOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                             local_sec_info_t &sec_info) {
  auto out_g_info = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto in_g_info = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  int b0_shape[MAX_SHAPE_DIMS] = {(int)in_g_info.n_slice, (int)c,
                                  (int)in_g_info.h_slice, (int)w};
  auto b1_val = (float)getConstVal().convertToDouble();
  auto if_relu = getDoRelu();
  auto relu_limit = getReluLimit().convertToDouble();
  int inversed = 1;
  BM1684::instance().dl_nodechip_const_binary_local(
      in_g_info.out_addr, (uint32_t *)b0_shape, b1_val, out_g_info.out_addr,
      BINARY_DIV, inversed, if_relu, relu_limit,
      (CMD_ID_NODE *)BM1684::instance()->bdc_node);
}

uint32_t tpu::ReciprocalOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::ReciprocalOp::get_fw_type_bm1684() { return -1; }

int32_t tpu::ReciprocalOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

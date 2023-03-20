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

void tpu::AddConstOp::codegen_global_bm1684() {
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  if (module::isUniformQuantized(getInput())) {
    auto b0_mulpiler = getMultiplier();
    auto b0_rshift = getRshift();
    int is_signs[3] = {module::isSign(getInput()), 1,
                       module::isSign(getOutput())};
    int is_int8s[3] = {1, 0, 1};
    int b0_shape[4] = {(int)n, (int)c, (int)h, (int)w};
    int16_t b1_val = (int16_t)getConstVal().convertToDouble();
    BM1684::instance().dl_nodechip_const_binary_fix8b_forward_parallel(
        module::getAddress(getInput()), module::getAddress(getOutput()), b1_val,
        b0_shape, 4, BINARY_ADD, b0_mulpiler, 0, b0_rshift, 0, 0, getDoRelu(),
        is_int8s, is_signs, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  } else {
    BM1684::instance().dl_nodechip_const_binary(
        module::getAddress(getInput()), n * c * h * w,
        getConstVal().convertToDouble(), module::getAddress(getOutput()),
        BINARY_ADD, 0, getDoRelu(), getReluLimit().convertToDouble(),
        (CMD_ID_NODE *)BM1684::instance().cmdid_node,
        module::getStorageType(getInput()).isa<IntegerType>());
  }
}

int64_t tpu::AddConstOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::AddConstOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                           local_sec_info_t &sec_info) {
  auto out_g_info = getGroupInfo(n_step, h_step);
  auto in_g_info = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  int b0_shape[MAX_SHAPE_DIMS] = {(int)in_g_info.n_slice, (int)c,
                                  (int)in_g_info.h_slice, (int)w};
  auto b1_val = (float)getConstVal().convertToDouble();
  auto if_relu = getDoRelu();
  auto relu_limit = getReluLimit().convertToDouble();
  if (module::isUniformQuantized(getOutput())) {
    auto b0_mul = getMultiplier();
    auto b0_rshift = getRshift();
    int is_signs[3] = {module::isSign(getInput()), 1,
                       module::isSign(getOutput())};
    int is_int8s[3] = {1, 0, 1};
    BM1684::instance().dl_nodechip_const_binary_fix8b_forward_local(
        in_g_info.out_addr, out_g_info.out_addr, 0, b1_val, b0_shape, 4,
        BINARY_ADD, b0_mul, 0, b0_rshift, 0, 0, getDoRelu(), is_int8s, is_signs,
        BM1684::instance().bdc_node);
  } else {
    BM1684::instance().dl_nodechip_const_binary_local(
        in_g_info.out_addr, (uint32_t *)b0_shape, b1_val, out_g_info.out_addr,
        BINARY_ADD, 0, if_relu, relu_limit,
        (CMD_ID_NODE *)BM1684::instance().bdc_node);
  }
}

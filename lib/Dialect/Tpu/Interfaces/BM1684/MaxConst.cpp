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

void tpu::MaxConstOp::codegen_global_bm1684() {
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  if (module::isUniformQuantized(getInput())) {
    llvm_unreachable("Not Implemented");
  } else {
    BM1684::instance().dl_nodechip_const_binary(
        module::getAddress(getInput()), n * c * h * w,
        getConstVal().convertToDouble(), module::getAddress(getOutput()),
        BINARY_MAX, 0, getDoRelu(), getReluLimit().convertToDouble(),
        (CMD_ID_NODE *)BM1684::instance().cmdid_node,
        module::getStorageType(getInput()).isa<IntegerType>());
  }
}

int64_t tpu::MaxConstOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::MaxConstOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                           local_sec_info_t &sec_info) {
  auto out_g_info = getGroupInfo(n_step, h_step);
  auto in_g_info = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  uint32_t bottom_shapes[MAX_SHAPE_DIMS] = {
      (u_int32_t)in_g_info.n_slice, (u_int32_t)c, (u_int32_t)in_g_info.h_slice,
      (u_int32_t)w};
  auto bottom1_val = getConstVal().convertToDouble();
  auto if_relu = getDoRelu();
  auto relu_limit = getReluLimit().convertToDouble();
  BM1684::instance().dl_nodechip_const_binary_local(
      in_g_info.out_addr, bottom_shapes, bottom1_val, out_g_info.out_addr,
      BINARY_MAX, 0, if_relu, relu_limit,
      (CMD_ID_NODE *)BM1684::instance().bdc_node);
}

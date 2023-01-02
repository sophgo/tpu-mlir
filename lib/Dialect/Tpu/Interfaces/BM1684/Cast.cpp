//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"

#include "tpu_mlir/Support/Module.h"



using namespace tpu_mlir::backend;

void tpu::CastOp::codegen_global_bm1684() {
  bool qInput = module::isUniformQuantized(getInput());
  bool qOutput = module::isUniformQuantized(getOutput());
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  if (qInput && !qOutput) {
    // int8 => fp32
    auto scale = module::getUniformQuantizedType(getInput()).getScale();
    BM1684::instance().dl_nodechip_global_int2float(
        module::getAddress(getInput()), module::getAddress(getOutput()), n, c, h, w,
        1, STORAGE_MODE_4N_INT8, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    BM1684::instance().dl_nodechip_const_binary(
        module::getAddress(getOutput()), n * c * h * w, scale,
        module::getAddress(getOutput()), BINARY_MUL, 0, 0, 0,
        (CMD_ID_NODE *)BM1684::instance().cmdid_node, 0);
  } else if (qOutput && !qInput) {
    // fp32 => int8
    auto scale = module::getUniformQuantizedType(getOutput()).getScale();
    BM1684::instance().dl_nodechip_const_binary(
        module::getAddress(getInput()), n * c * h * w, scale,
        module::getAddress(getInput()), BINARY_DIV, 0, 0, 0,
        (CMD_ID_NODE *)BM1684::instance().cmdid_node, 0);
    BM1684::instance().dl_nodechip_float2int8_v2(
        module::getAddress(getInput()), module::getAddress(getOutput()), n, c, h, w,
        1, STORAGE_MODE_4N_INT8, ROUND_INF,
        (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  } else {
    dump();
    llvm_unreachable("CastOp type error");
  }
}

int64_t tpu::CastOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {

  return 0;
}

void tpu::CastOp::codegen_local_bm1684(int64_t n_step, int64_t h_step) {
  llvm_unreachable("Not Implemented");
}

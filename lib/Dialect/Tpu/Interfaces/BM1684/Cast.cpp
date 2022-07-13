//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

void tpu::CastOp::codegen_global_int8_bm1684() {
  bool qInput = Quant::isUniformQuantized(input());
  bool qOutput = Quant::isUniformQuantized(output());
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  if (qInput && !qOutput) {
    // int8 => fp32
    auto scale = Quant::getUniformQuantizedType(input()).getScale();
    BM1684::instance().dl_nodechip_global_int2float(
        Module::getAddress(input()), Module::getAddress(output()), n, c, h, w,
        1, STORAGE_MODE_4N_INT8, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    BM1684::instance().dl_nodechip_const_binary(
        Module::getAddress(output()), n * c * h * w, scale,
        Module::getAddress(output()), BM_BINARY_MUL, 0, 0, 0,
        (CMD_ID_NODE *)BM1684::instance().cmdid_node, 0);
  } else if (qOutput && !qInput) {
    // fp32 => int8
    auto scale = Quant::getUniformQuantizedType(output()).getScale();
    BM1684::instance().dl_nodechip_const_binary(
        Module::getAddress(input()), n * c * h * w, scale,
        Module::getAddress(input()), BM_BINARY_DIV, 0, 0, 0,
        (CMD_ID_NODE *)BM1684::instance().cmdid_node, 0);
    BM1684::instance().dl_nodechip_float2int8_v2(
        Module::getAddress(input()), Module::getAddress(output()), n, c, h, w,
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

void tpu::CastOp::codegen_local_int8_bm1684(int64_t n_step, int64_t h_step) {
  llvm_unreachable("support later");
}

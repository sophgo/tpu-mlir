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

void tpu::AvgPoolOp::codegen_global_int8_bm1684() {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool is_global, count_include_pad, relu;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);
  BM1684::instance().dl_nodechip_pooling_fix8b_forward_parallel_with_data_split(
      Module::getAddress(input()), Module::getAddress(output()), n, c, ih, iw,
      kh, kw, pt, pb, pl, pr, sh, sw, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
      relu ? 1 : 0, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
}

void tpu::MaxPoolOp::codegen_global_int8_bm1684() {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool is_global, count_include_pad, relu;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);
  BM1684::instance().dl_nodechip_pooling_fix8b_forward_parallel_with_data_split(
      Module::getAddress(input()), Module::getAddress(output()), n, c, ih, iw,
      kh, kw, pt, pb, pl, pr, sh, sw, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
      relu ? 1 : 0, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
}

int64_t tpu::AvgPoolOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                             int64_t out_lmem_bytes) {
  return 0;
}

int64_t tpu::MaxPoolOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                             int64_t out_lmem_bytes) {
  return 0;
}

void tpu::AvgPoolOp::codegen_local_int8_bm1684(int64_t n_step, int64_t h_step) {
  llvm_unreachable("support later");
}

void tpu::MaxPoolOp::codegen_local_int8_bm1684(int64_t n_step, int64_t h_step) {
  llvm_unreachable("support later");
}

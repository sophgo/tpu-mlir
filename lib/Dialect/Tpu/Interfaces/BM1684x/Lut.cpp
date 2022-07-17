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
#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned long long input_addr;
  unsigned long long table_addr;
  unsigned long long output_addr;
  unsigned int buffer_addr; // used only for local layer
  int shape[MAX_SHAPE_DIMS];
  int shape_dim;
  int table_length;
  int input_dtype;
  int table_dtype;
  int output_dtype;
  int is_local_layer;
} lut_param_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

void tpu::LutOp::codegen_global_int8_bm1684x() {
  lut_param_t p = {0};
  p.input_addr = Module::getAddress(input());
  p.table_addr = Module::getAddress(table());
  p.output_addr = Module::getAddress(output());
  p.input_dtype = BM1684x::getDataType(input());
  p.table_dtype = BM1684x::getDataType(table());
  p.output_dtype = BM1684x::getDataType(output());
  p.table_length = 256;
  p.is_local_layer = 0;
  p.shape_dim = 4;
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  p.shape[0] = n;
  p.shape[1] = c;
  p.shape[2] = h;
  p.shape[3] = w;
  BM1684x::instance().call_global_func("backend_api_lut", &p, sizeof(p));
}

void tpu::LutOp::codegen_global_float_bm1684x() {
  llvm_unreachable("Codegen to be supported");
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::LutOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  return 0;
}

void tpu::LutOp::codegen_local_int8_bm1684x(int64_t n_step, int64_t h_step) {
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto table_gi = LocalGenInterface::getGroupInfo(table(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  lut_param_t p = {0};
  p.input_addr = in_gi.out_addr;
  p.table_addr = table_gi.out_addr;
  p.output_addr = gi.out_addr;
  p.input_dtype = BM1684x::getDataType(input());
  p.table_dtype = BM1684x::getDataType(table());
  p.output_dtype = BM1684x::getDataType(output());
  p.table_length = 256;
  p.is_local_layer = 1;
  p.shape_dim = 4;
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  p.shape[0] = gi.n_slice;
  p.shape[1] = c;
  p.shape[2] = gi.h_slice;
  p.shape[3] = w;
  BM1684x::instance().call_local_func("backend_api_lut", &p, sizeof(p));
}

void tpu::LutOp::codegen_local_float_bm1684x(int64_t n_step, int64_t h_step) {
  llvm_unreachable("support later");
}

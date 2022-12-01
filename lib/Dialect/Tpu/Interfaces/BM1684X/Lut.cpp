//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
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

void tpu::LutOp::codegen_global_bm1684x() {
  lut_param_t p = {0};
  p.input_addr = Module::getAddress(input());
  p.table_addr = Module::getAddress(table());
  p.output_addr = Module::getAddress(output());
  p.input_dtype = BM168x::getDataType(input());
  p.table_dtype = BM168x::getDataType(table());
  p.output_dtype = BM168x::getDataType(output());
  p.table_length = 256;
  p.is_local_layer = 0;
  p.shape_dim = 4;
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  p.shape[0] = n;
  p.shape[1] = c;
  p.shape[2] = h;
  p.shape[3] = w;
  auto op = getOperation();
  BM168x::call_global_func("backend_api_lut", &p, sizeof(p));
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

void tpu::LutOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto table_gi = LocalGenInterface::getGroupInfo(table(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  lut_param_t p = {0};
  p.input_addr = in_gi.out_addr;
  p.table_addr = table_gi.out_addr;
  p.output_addr = gi.out_addr;
  p.input_dtype = BM168x::getDataType(input());
  p.table_dtype = BM168x::getDataType(table());
  p.output_dtype = BM168x::getDataType(output());
  p.table_length = 256;
  p.is_local_layer = 1;
  p.shape_dim = 4;
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  p.shape[0] = gi.n_slice;
  p.shape[1] = c;
  p.shape[2] = gi.h_slice;
  p.shape[3] = w;
  auto op = getOperation();
  BM168x::call_local_func("backend_api_lut", &p, sizeof(p));
}

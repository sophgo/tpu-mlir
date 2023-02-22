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

#include "tpu_mlir/Support/Module.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint64_t input_addr;
  uint64_t table_addr;
  uint64_t output_addr;
  unsigned int buffer_addr; // used only for local layer
  int shape[MAX_SHAPE_DIMS];
  int shape_dim;
  int table_length;
  int input_dtype;
  int table_dtype;
  int output_dtype;
  int is_local_layer;
} scalelut_param_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

using namespace tpu_mlir::backend;

void tpu::ScaleLutOp::codegen_global_bm1684x() {
  scalelut_param_t p = {0};
  p.input_addr = module::getAddress(getInput());
  p.table_addr = module::getAddress(getTable());
  p.output_addr = module::getAddress(getOutput());
  p.input_dtype = BM168x::getDataType(getInput());
  p.table_dtype = BM168x::getDataType(getTable());
  p.output_dtype = BM168x::getDataType(getOutput());
  p.is_local_layer = 0;
  p.shape_dim = 4;
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  p.shape[0] = n;
  p.shape[1] = c;
  p.shape[2] = h;
  p.shape[3] = w;
  p.table_length = 256;
  BM168x::call_global_func("backend_api_scalelut", &p, sizeof(p));
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ScaleLutOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}


void tpu::ScaleLutOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step, local_sec_info_t &sec_info) {
  llvm_unreachable("Not Implemented");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================

int64_t tpu::ScaleLutOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

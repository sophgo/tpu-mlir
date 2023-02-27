//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;


void tpu::SliceOp::codegen_global_bm1684x() {
  auto p = parseParam();
  if (p.fusible) {
    return;
  }
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::fix_shape(input_spec->at(0), p.is_4);
  BM168x::fix_shape(output_spec->at(0), p.os_4);
  strideslice_common_spec_t param = {0};
  param.begin_mask = 0;
  param.end_mask = 0;
  int num_dims = p.is_4.size();
  for (int i = 0; i < num_dims; i++) {
    param.begin_index[i] = p.offset_4[i];
    param.end_index[i] = p.os_4[i] * p.step_4[i] + p.offset_4[i];
    param.strides[i] = p.step_4[i];
  }
  BM168x::call_global_func("backend_api_strideslice_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::SliceOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  llvm_unreachable("Not support.");
  return 0;
}

void tpu::SliceOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  llvm_unreachable("Not support.");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::SliceOp::dyn_codegen_global_bm1684x(void *buffer) { return 0; }

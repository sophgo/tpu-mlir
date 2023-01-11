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

// =========================================
// GlobalGenInterface
// =========================================

void tpu::ClipOp::codegen_global_bm1684x() {
  llvm_unreachable("Not Implemented");
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ClipOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                           int64_t out_lmem_bytes,
                                           int64_t in_nslice, int64_t in_hslice,
                                           int64_t out_nslice,
                                           int64_t out_hslice) {
  llvm_unreachable("Not Implemented");
  return 0;
}

void tpu::ClipOp::assign_sec_info(int64_t n_step, int64_t h_step,
                                  void *sec_info_) {
  llvm_unreachable("Not Implemented");
}

void tpu::ClipOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                        void *sec_info_) {
  llvm_unreachable("Not Implemented");
}

//dynamic codegen
int64_t tpu::ClipOp::dyn_codegen_local_bm1684x(void *buffer) {
return 0;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ClipOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

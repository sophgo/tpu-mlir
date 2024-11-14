//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

// =========================================
// GlobalGenInterface
// =========================================

void tpu::RequantIntOp::codegen_global_cv18xx(int64_t layer_id) {
  llvm_unreachable("Not supported now");
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::RequantIntOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  llvm_unreachable("Not supported now");
  return 0;
}

void tpu::RequantIntOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                             int64_t d_step, int64_t w_step,
                                             group_type_t group_type,
                                             local_sec_info_t &sec_info,
                                             int64_t layer_id) {
  llvm_unreachable("Not supported now");
}

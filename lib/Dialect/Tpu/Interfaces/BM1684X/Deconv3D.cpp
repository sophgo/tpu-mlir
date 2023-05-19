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
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Dnnl/Deconv.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

// ======================================
// GlobalGenInterface
// ======================================

void tpu::Deconv3DOp::codegen_global_bm1684x() {
  llvm_unreachable("Not Implemented");
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::Deconv3DOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  // TODO
  return 0;
}

void tpu::Deconv3DOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                            int64_t h_step, int64_t d_step,
                                            int64_t w_step,
                                            group_type_t group_type,
                                            local_sec_info_t &sec_info) {
  llvm_unreachable("Not Implemented");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::Deconv3DOp::dyn_codegen_global_bm1684x(void *buffer) {
  llvm_unreachable("Not Implemented");
}

int64_t tpu::Deconv3DOp::get_fw_type_bm1684x() { return -1; }

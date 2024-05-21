//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::LutBF16Op::codegen_global_bm1684x() {
  UNREACHABLE_THIS("Not Implemented");
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::LutBF16Op::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return 0;
}

void tpu::LutBF16Op::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                           int64_t h_step, int64_t d_step,
                                           int64_t w_step,
                                           group_type_t group_type,
                                           local_sec_info_t &sec_info) {
  UNREACHABLE_THIS("Not Implemented");
}

// dynamic codegen
int64_t tpu::LutBF16Op::dyn_codegen_local_bm1684x(void *buffer) { return 0; }

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::LutBF16Op::dyn_codegen_global_bm1684x(void *buffer) { return 0; }

int64_t tpu::LutBF16Op::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }

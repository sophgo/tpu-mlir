//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

void tpu::WhereBnbwdOp::codegen_global_bm1684() {
  UNREACHABLE_THIS("Not Implemented");
}

// int64_t tpu::WhereBnbwdOp::getBufferSize_bm1684(
//     int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
//     int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
//   llvm_unreachable("Not supported now");
//   return 0;
// }

// void tpu::WhereBnbwdOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
//                                               local_sec_info_t &sec_info) {
//   llvm_unreachable("Not supported now");
// }

// ======================================
// Dynamic GlobalGenInterface
// ======================================

uint32_t tpu::WhereBnbwdOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::WhereBnbwdOp::get_fw_type_bm1684() { return -1; }

// ======================================
// Dynamic LocalGenInterface
// ======================================

// int32_t tpu::WhereBnbwdOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
//   UNREACHABLE_THIS("Not Implemented");
//   return 0;
// }

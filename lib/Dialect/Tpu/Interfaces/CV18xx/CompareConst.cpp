//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::CompareConstOp::codegen_global_cv18xx(int64_t layer_id) {
  gaddr_t input_gaddr = module::getAddress(this->getInput());
  gaddr_t output_gaddr = module::getAddress(this->getOutput());
  // parseparam
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w, false);
  auto fmt =
      module::isUniformQuantized(getOutput()) ? CVK_FMT_U8 : CVK_FMT_BF16;
  // auto pos = positive();
  auto pos = false;            // todo
  assert(fmt == CVK_FMT_BF16); // todo
  cvi_backend_zero_mask_kernel(layer_id, input_gaddr, output_gaddr, n, c, h, w,
                               pos, fmt);
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::CompareConstOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  llvm_unreachable("Not supported now");
}

void tpu::CompareConstOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                               int64_t d_step, int64_t w_step,
                                               group_type_t group_type,
                                               local_sec_info_t &sec_info,
                                               int64_t layer_id) {
  llvm_unreachable("Not supported now");
}

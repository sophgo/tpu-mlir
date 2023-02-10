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
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

void tpu::PixelNormOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  const bool have_weight = !getWeight().getType().isa<NoneType>();
  const bool have_bias = !getBias().getType().isa<NoneType>();
  pixel_norm_global_spec_t param = {0};
  param.common.eps = getEps().convertToDouble();
  param.common.affine = (have_weight << 0) + (have_bias << 1);
  BM168x::call_global_func("backend_api_pixel_norm_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::PixelNormOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::PixelNormOp::codegen_local_bm1684x(int64_t n_step,
                                             int64_t h_step,
                                             local_sec_info_t &sec_info) {
  llvm_unreachable("Not supported now");
}


// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::PixelNormOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

// ======================================
// Dynamic LocalGenInterface
// ======================================
int64_t tpu::PixelNormOp::dyn_codegen_local_bm1684x(void *buffer) {
  return 0;
}

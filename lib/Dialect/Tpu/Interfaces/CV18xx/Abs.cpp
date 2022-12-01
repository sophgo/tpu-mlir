//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
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

// int8
void tpu::AbsOp::codegen_global_cv18xx(int64_t layer_id) {
  int input_num = 1;
  gaddr_t input = Module::getAddress(this->input());
  gaddr_t ga_inputs[] = {input};
  int64_t n, c, h, w;
  Module::getNCHW(this->input(), n, c, h, w);
  gaddr_t ga_output = Module::getAddress(output());
  bool do_relu = false;
  bool do_early_stride = false;
  int early_stride_h = 0;
  int early_stride_w = 0;
  if (Quant::isUniformQuantized(output())) {
    cvi_backend_tg_eltwise_abs_kernel(layer_id, ga_inputs, ga_output, input_num, n, c, h, w,
                                      do_relu, do_early_stride, early_stride_h, early_stride_w,
                                      0, NULL, NULL, CVK_FMT_I8);
  } else {
    cvi_backend_tg_eltwise_abs_kernel(layer_id, ga_inputs, ga_output, input_num, n, c, h, w,
                                      do_relu, do_early_stride, early_stride_h, early_stride_w,
                                      0, NULL, NULL, CVK_FMT_BF16);
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::AbsOp::getBufferSize_cv18xx(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes,
                                         int64_t in_nslice, int64_t in_hslice,
                                         int64_t out_nslice,
                                         int64_t out_hslice) {
  llvm_unreachable("Not supported now");
}

void tpu::AbsOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step) {
  llvm_unreachable("Not supported now");
}

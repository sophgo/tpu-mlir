//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Conv.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

// ======================================
// WeightReorderInterface
// ======================================

template <typename T>
static void filter_reorder(std::shared_ptr<std::vector<T>> &filter,
                           std::vector<int64_t> &shape) {
  llvm_unreachable("Not supported now");
}

// void tpu::Conv3DOp::weight_reorder_int8_cv18xx() {
//   llvm_unreachable("Not supported now");
// }

// void tpu::Conv3DOp::weight_reorder_bf16_cv18xx() {
//   llvm_unreachable("Not supported now");
// }

void tpu::Conv3DOp::codegen_global_cv18xx( int64_t layer_id) {
  llvm_unreachable("Not supported now");
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::Conv3DOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  llvm_unreachable("Not supported now");
  return 0;
}

void tpu::Conv3DOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step) {
   llvm_unreachable("Not supported now");
}

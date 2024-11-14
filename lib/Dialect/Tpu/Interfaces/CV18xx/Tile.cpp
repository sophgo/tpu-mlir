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
void tpu::TileOp::codegen_global_cv18xx(int64_t layer_id) {
  auto tile = module::getI64Array(getTile());
  int count = 0;
  int index = 0;
  for (int i = 0; i < tile->size(); ++i) {
    if (tile->at(i) > 1) {
      count++;
      index = i;
    }
  }
  if (count > 1) {
    llvm_unreachable("Only support tile one dimension");
  }
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  int64_t n, c, h, w;
  auto fmt =
      module::isUniformQuantized(getOutput()) ? CVK_FMT_I8 : CVK_FMT_BF16;
  module::getNCHW(getInput(), n, c, h, w);

  cvi_backend_tg_tile_kernel(layer_id, ga_input, ga_output, n, c, h, w, index,
                             tile->at(index), fmt);
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::TileOp::getBufferSize_cv18xx(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  return 0;
}

void tpu::TileOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                       int64_t d_step, int64_t w_step,
                                       group_type_t group_type,
                                       local_sec_info_t &sec_info,
                                       int64_t layer_id) {
  llvm_unreachable("Not supported now");
}

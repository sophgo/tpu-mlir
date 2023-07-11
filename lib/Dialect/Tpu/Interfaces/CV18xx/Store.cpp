//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace tpu_mlir::backend;

void tpu::StoreOp::codegen_global_cv18xx(int64_t layer_id) {
  llvm_unreachable("not support now");
}

int64_t tpu::StoreOp::getBufferSize_cv18xx(int64_t in_lmem_bytes,
                                           int64_t out_lmem_bytes,
                                           int64_t in_nslice, int64_t in_hslice,
                                           int64_t out_nslice,
                                           int64_t out_hslice) {
  return 0;
}

void tpu::StoreOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                        int64_t d_step, int64_t w_step,
                                        group_type_t group_type,
                                        local_sec_info_t &sec_info,
                                        int64_t layer_id) {

  std::vector<int64_t> shape;
  int64_t g_n, g_c, g_h, g_w;
  module::getNCHW(getOutput(), g_n, g_c, g_h, g_w); // global
  cvk_tg_shape_t gshape = {(uint32_t)g_n, (uint32_t)g_c, (uint32_t)g_h,
                           (uint32_t)g_w};
  auto ifmt = CV18xx::getDataType(getInput());
  auto ofmt = CV18xx::getDataType(getOutput());
  auto g_stride = CV18xx::tg_default_stride(gshape, ofmt);
  auto gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  bool transpose = false;
  bool isNeuron = true;

  int64_t g_offset = (gi.n_idx * g_stride.n + gi.h_idx * g_stride.h);
  gaddr_t g_addr = module::getAddress(getOutput()) + g_offset;

  assert((ifmt == CVK_FMT_BF16 || ifmt == CVK_FMT_I8) &&
         (ofmt == CVK_FMT_BF16 || ofmt == CVK_FMT_I8) &&
         "current store only support int8/bf16");

  cvi_backend_tl_store_stride(layer_id, g_addr, gi.out_addr, gi.n_slice, g_c,
                              gi.h_slice, g_w, g_c, g_h, g_w, transpose,
                              gi.eu_align, isNeuron, ifmt, ofmt);
}

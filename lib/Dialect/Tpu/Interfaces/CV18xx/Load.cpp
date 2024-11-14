//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::LoadOp::codegen_global_cv18xx(int64_t layer_id) {
  llvm_unreachable("global not support");
}

int64_t tpu::LoadOp::getBufferSize_cv18xx(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  return 0;
}

void tpu::LoadOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                       int64_t d_step, int64_t w_step,
                                       group_type_t group_type,
                                       local_sec_info_t &sec_info,
                                       int64_t layer_id) {
  auto gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  assert(false == gi.overstepped);
  int64_t g_n, g_c, g_h, g_w;
  module::getNCHW(getOutput(), g_n, g_c, g_h, g_w);
  cvk_tg_shape_t gshape = {(uint32_t)g_n, (uint32_t)g_c, (uint32_t)g_h,
                           (uint32_t)g_w};
  auto ifmt = CV18xx::getDataType(getInput());
  auto ofmt = CV18xx::getDataType(getOutput());
  auto g_stride = CV18xx::tg_default_stride(gshape, ifmt);
  bool bcompressed = false;
  bool transpose = false;
  bool isNeuron = true;
  if (module::isWeight(module::getOriValue(getInput()))) {
    isNeuron = false;
  }

  if (isNeuron) {
    if (ifmt == CVK_FMT_U8) {
      ifmt = CVK_FMT_I8;
    }
    if (ofmt == CVK_FMT_U8) {
      ofmt = CVK_FMT_I8;
    }
    assert((ifmt == CVK_FMT_BF16 || ifmt == CVK_FMT_I8) &&
           (ofmt == CVK_FMT_BF16 || ofmt == CVK_FMT_I8) &&
           "current load neuron only support int8/bf16");
  } else {
    assert(
        (ofmt == CVK_FMT_BF16 || ofmt == CVK_FMT_I8 || ofmt == CVK_FMT_U16) &&
        "current load weight only support int8/uint16/bf16");
    if (ofmt == CVK_FMT_U16) {
      ofmt = CVK_FMT_BF16;
    }
    ifmt = ofmt;
  }

  int64_t g_offset = (gi.n_idx * g_stride.n + gi.h_idx * g_stride.h);
  gaddr_t src_gaddr = module::getAddress(getInput()) + g_offset;

  if (getDoBcast() == true) {
    cvi_backend_tl_load_stride_broadcast(
        layer_id, src_gaddr, gi.out_addr, gi.n_slice, g_c, gi.h_slice, g_w, g_c,
        g_h, g_w, gi.eu_align, isNeuron, ifmt, ofmt, bcompressed);
  } else {
    cvi_backend_tl_load_stride(layer_id, src_gaddr, gi.out_addr, gi.n_slice,
                               g_c, gi.h_slice, g_w, g_c, g_h, g_w, transpose,
                               gi.eu_align, isNeuron, ifmt, ofmt, bcompressed);
  }
}

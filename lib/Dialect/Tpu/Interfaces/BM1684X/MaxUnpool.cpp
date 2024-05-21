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
void tpu::MaxUnpoolOp::codegen_global_bm1684x() {
  assert(getScaleH() == getScaleW());
  int64_t in, ic, ih, iw;
  module::getNCHW(getInput(), in, ic, ih, iw);
  int64_t on, oc, oh, ow;
  module::getNCHW(getOutput(), on, oc, oh, ow);

  upsamplemask_param_t spec = {0};
  spec.bottom_global_offset = module::getAddress(getInput());
  spec.bottom_mask_global_offset = module::getAddress(getMask());
  spec.top_global_offset = module::getAddress(getOutput());
  spec.bottom_global_N = in;
  spec.bottom_c = ic;
  spec.bottom_h = ih;
  spec.bottom_w = iw;
  spec.top_c = oc;
  spec.top_h = oh;
  spec.top_w = ow;
  BM168x::call_global_func("backend_api_upsamplemask_global", &spec,
                           sizeof(spec));
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::MaxUnpoolOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return 0;
}

void tpu::MaxUnpoolOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                             int64_t h_step, int64_t d_step,
                                             int64_t w_step,
                                             group_type_t group_type,
                                             local_sec_info_t &sec_info) {
  UNREACHABLE_THIS("Not Implemented");
}

// dynamic codegen
int64_t tpu::MaxUnpoolOp::dyn_codegen_local_bm1684x(void *buffer) { return 0; }

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::MaxUnpoolOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_upsamplemask_global_spec_t);
  assert(getScaleH() == getScaleW());
  int64_t on, oc, oh, ow;
  module::getNCHW(getOutput(), on, oc, oh, ow);
  dyn_upsamplemask_global_spec_t spec = {0};
  spec.common.top_c = oc;
  spec.common.top_h = oh;
  spec.common.top_w = ow;
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::MaxUnpoolOp::get_fw_type_bm1684x() {
  return FW_BMNET_UPSAMPLEMASK;
}

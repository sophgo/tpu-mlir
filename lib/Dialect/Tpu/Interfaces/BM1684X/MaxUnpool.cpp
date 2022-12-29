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




using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned long long bottom_global_offset;
  unsigned long long bottom_mask_global_offset;
  unsigned long long top_global_offset;
  int bottom_global_N;
  int bottom_c;
  int bottom_h;
  int bottom_w;
  int top_c;
  int top_h;
  int top_w;
} upsamplemask_param_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================
void tpu::MaxUnpoolOp::codegen_global_bm1684x() {
  assert(scale_h() == scale_w());
  int64_t in, ic, ih, iw;
  module::getNCHW(input(), in, ic, ih, iw);
  int64_t on, oc, oh, ow;
  module::getNCHW(output(), on, oc, oh, ow);

  upsamplemask_param_t spec = {0};
  spec.bottom_global_offset = module::getAddress(input());
  spec.bottom_mask_global_offset = module::getAddress(mask());
  spec.top_global_offset = module::getAddress(output());
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
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::MaxUnpoolOp::assign_sec_info(int64_t n_step, int64_t h_step,
                                             void *sec_info_) {
  llvm_unreachable("Not Implemented");
}

void tpu::MaxUnpoolOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                             void *sec_info_) {
  llvm_unreachable("Not Implemented");
}

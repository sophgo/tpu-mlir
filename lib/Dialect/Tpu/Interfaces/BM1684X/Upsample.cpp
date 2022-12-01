//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int size;
  int if_relu;
} upsample_spec_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================
void tpu::UpsampleOp::codegen_global_bm1684x() {
  assert(scale_h() == scale_w());
  auto op = getOperation();
  // int64_t n, c, h, w;
  // Module::getNCHW(input(), n, c, h, w);

  upsample_spec_t spec = {0};
  spec.size = scale_h();
  spec.if_relu = do_relu();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_upsample_global", &spec,
                                       sizeof(spec), input_spec->data(),
                                       output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::UpsampleOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::UpsampleOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  assert(scale_h() == scale_w());
  auto op = getOperation();
  int64_t n, c, h, w, oh, ow;
  Module::getNCHW(input(), n, c, h, w);
  Module::getNCHW(output(), n, c, oh, ow);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  upsample_spec_t spec = {0};
  spec.size = scale_h();
  spec.if_relu = do_relu();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  local_sec_info_t sec_info;
  memset(&sec_info, 0, sizeof(sec_info));
  sec_info.n_slice = in_gi.n_slice;
  sec_info.h_slice = in_gi.h_slice;
  sec_info.h_idx = in_gi.h_idx;
  sec_info.is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == h);
  sec_info.is_w_split = false;
  sec_info.w_slice = w;

  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_slice = ow;
  BM168x::call_local_func("backend_api_upsample_local", &spec,
                                      sizeof(spec), &sec_info, input_spec->data(),
                                      output_spec->data());
}

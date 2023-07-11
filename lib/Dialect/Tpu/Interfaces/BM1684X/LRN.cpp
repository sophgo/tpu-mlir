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
void tpu::LRNOp::codegen_global_bm1684x() {
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  lrn_global_param_t p = {0};
  p.input_addr = module::getAddress(getInput());
  p.output_addr = module::getAddress(getOutput());
  p.size = getSize();

  p.input_n = n;
  p.input_c = c;
  p.input_h = h;
  p.input_w = w;

  p.alpha = getAlpha().convertToDouble();
  p.beta = getBeta().convertToDouble();
  p.k = getBias().convertToDouble();

  p.dtype = BM168x::getDataType(getInput());
  BM168x::call_global_func("backend_api_lrn_global", &p,
                           sizeof(lrn_global_param_t));
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::LRNOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_lrn_global_param_t);
  dyn_lrn_global_param_t p = {0};
  p.common.size = getSize();
  p.common.alpha = getAlpha().convertToDouble();
  p.common.beta = getBeta().convertToDouble();
  p.common.k = getBias().convertToDouble();
  p.common.dtype = BM168x::getDataType(getInput());
  return BM168x::dynamic_spec_to_buffer(buffer, p);
}

int64_t tpu::LRNOp::get_fw_type_bm1684x() { return FW_BMNET_LRN; }

int64_t tpu::LRNOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  llvm_unreachable("not supported now");
  return 0;
}

void tpu::LRNOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                       int64_t h_step, int64_t d_step,
                                       int64_t w_step, group_type_t group_type,
                                       local_sec_info_t &sec_info) {
  llvm_unreachable("not supported now");
}

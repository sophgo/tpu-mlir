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
#include "tpu_mlir/Dialect/Tpu/Transforms/DynCompileCommon.hpp"
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
  if (!buffer) return sizeof(dyn_lrn_global_param_t);
  dyn_lrn_global_param_t p = {0};
  p.common.size = getSize();
  p.common.alpha = getAlpha().convertToDouble();
  p.common.beta = getBeta().convertToDouble();
  p.common.k = getBias().convertToDouble();
  p.common.dtype = BM168x::getDataType(getInput());
  return BM168x::dynamic_spec_to_buffer(buffer, p);
}

int64_t tpu::LRNOp::get_layer_type() {
  return FW_BMNET_LRN;
}

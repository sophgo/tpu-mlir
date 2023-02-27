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
  return 0;
}

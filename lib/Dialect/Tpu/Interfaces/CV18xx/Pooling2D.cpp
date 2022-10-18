//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Dnnl/Pool.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::backend;
using namespace tpu_mlir::helper;
// using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::Pool2DOp::codegen_global_cv18xx(void* ctx, int64_t layer_id) {
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  pool_attr_t attrs = {0};
  parseParam(&attrs);
  if (pool_mode() == tpu::PoolMode::Avg) {
    assert(!attrs.do_relu);
    gaddr_t ga_input = Module::getAddress(input());
    gaddr_t ga_output = Module::getAddress(output());

    cvi_backend_tg_fixed_avg_pooling_kernel(
    *backend_ctx,
    layer_id, // layer_id,
    ga_input,            // input_data_gaddr,
    ga_output,           // output_data_gaddr,
    attrs.n, attrs.c, attrs.ih, attrs.iw,
    attrs.kh, attrs.kw,
    attrs.pad_h, attrs.pad_h_after,
    attrs.pad_w, attrs.pad_w_after, // pad (t, b, l, r)
    attrs.sh, attrs.sw,
    attrs.do_relu,        // int do_relu,
    (int8_t)rshift().value(),       // int right_shift_width,
    (int8_t)multiplier().value(),   // &threshold_x_quantized,
    true);

  } else {
    llvm_unreachable("Not supported now");
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::Pool2DOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::Pool2DOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step) {
  llvm_unreachable("Not supported now");
}

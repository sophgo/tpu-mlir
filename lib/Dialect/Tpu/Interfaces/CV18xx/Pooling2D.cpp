//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Pool.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"


using namespace tpu_mlir::backend;


// =========================================
// GlobalGenInterface
// =========================================
void tpu::Pool2DOp::codegen_global_cv18xx(int64_t layer_id) {
  auto attr = parseParam();
  assert(!attr.do_relu);
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  if (getPoolMode() == tpu::PoolMode::Avg) {
    if (module::isUniformQuantized(getOutput())) {
      cvi_backend_tg_fixed_avg_pooling_kernel(
          layer_id,  // layer_id,
          ga_input,  // input_data_gaddr,
          ga_output, // output_data_gaddr,
          attr.n, attr.c, attr.ih, attr.iw, attr.kh, attr.kw, attr.pad_h,
          attr.pad_h_after, attr.pad_w, attr.pad_w_after, // pad (t, b, l, r)
          attr.sh, attr.sw,
          attr.do_relu,                 // int do_relu,
          (int8_t)getRshift().value(),     // int right_shift_width,
          (int8_t)getMultiplier().value(), // &threshold_x_quantized,
          true);
    } else {
      cvi_backend_tg_bf16_pooling_kernel(

          layer_id,   // layer_id,
          ga_input,   // input_data_gaddr,
          ga_output,  // output_data_gaddr,
          GA_INVALID, // index_data_gaddr,
          GA_INVALID, // o_findex_data_gaddr,
          attr.n, attr.c, attr.ih, attr.iw, attr.kh, attr.kw, attr.pad_h,
          attr.pad_h_after, attr.pad_w, attr.pad_w_after, // pad (t, b, l, r)
          attr.sh, attr.sw,
          1,            // is_avg_pooling,
          0.0f,         // float avg_const,  // default(passing 0.0f) is 1/kh*kw
          attr.do_relu, // int do_relu,
          true);
    }
  } else if (getPoolMode() == tpu::PoolMode::Max) {
    if (module::isUniformQuantized(getOutput())) {
      cvi_backend_tg_fixed_max_pooling_kernel(

          layer_id,  // layer_id,
          ga_input,  // input_data_gaddr,
          ga_output, // output_data_gaddr,
          attr.n, attr.c, attr.ih, attr.iw, attr.kh, attr.kw, attr.pad_h,
          attr.pad_h_after, attr.pad_w, attr.pad_w_after, // pad (t, b, l, r)
          attr.sh, attr.sw,
          attr.do_relu, // int do_relu,
          true);
    } else {
      cvi_backend_tg_bf16_pooling_kernel(

          layer_id,   // layer_id,
          ga_input,   // input_data_gaddr,
          ga_output,  // output_data_gaddr,
          GA_INVALID, // index_data_gaddr,
          GA_INVALID, // o_findex_data_gaddr,
          attr.n, attr.c, attr.ih, attr.iw, attr.kh, attr.kw, attr.pad_h,
          attr.pad_h_after, attr.pad_w, attr.pad_w_after, // pad (t, b, l, r)
          attr.sh, attr.sw,
          0,            // is_avg_pooling,
          0.0f,         // float avg_const,  // default(passing 0.0f) is 1/kh*kw
          attr.do_relu, // int do_relu,
          true);
    }
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

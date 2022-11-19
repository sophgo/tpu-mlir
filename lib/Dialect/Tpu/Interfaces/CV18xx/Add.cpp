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
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::AddOp::codegen_global_cv18xx(int64_t layer_id) {
  int input_num = inputs().size();
  assert(input_num == 2);
  int64_t n, c, h, w;
  std::vector<gaddr_t> ga_inputs;
  for (int i = 0; i < input_num; ++i) {
    ga_inputs.emplace_back(Module::getAddress(inputs()[i]));
  }
  gaddr_t ga_output = Module::getAddress(output());

  bool do_early_stride = false;
  int early_stride_h = 0;
  int early_stride_w = 0;

  Module::getNCHW(output(), n, c, h, w);
  if (Quant::isUniformQuantized(output())) {
    auto multiplier_v = Module::getI64Array(multipliers(), input_num, 1);
    auto rshift_v = Module::getI64Array(rshifts(), 1, 0);
    int32_t rshift_int = static_cast<int32_t>(rshift_v->at(0));
    std::vector<int32_t> multiplier_int;
    for (int i = 0; i < input_num; ++i) {
      multiplier_int.emplace_back(multiplier_v->at(i));
    }
    std::vector<int> coeffs(input_num, 1);
    cvi_backend_tg_fixed_eltwise_add_kernel(
        layer_id, ga_inputs.data(), ga_output, input_num, n, c, h, w, do_relu(),
        do_early_stride, early_stride_h, early_stride_w, rshift_int,
        multiplier_int.data(), coeffs.data());
  } else {
    // TODO do_early_stride, coeffs
    std::vector<float> coeffs(input_num, 1.0);
    cvi_backend_tg_bf16_eltwise_add_kernel(
        layer_id,         // layer_id
        ga_inputs.data(), // gaddr_t ga_input[]
        ga_output,        // gaddr_t ga_output
        input_num,        // int input_size
        n, c, h, w,
        do_relu(), // bool do_relu
        do_early_stride, early_stride_h, early_stride_w, coeffs.data());
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::AddOp::getBufferSize_cv18xx(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes,
                                         int64_t in_nslice, int64_t in_hslice,
                                         int64_t out_nslice,
                                         int64_t out_hslice) {
  llvm_unreachable("Not supported now");
  return 0;
}

void tpu::AddOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step) {
  llvm_unreachable("Not supported now");
}

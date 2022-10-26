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
#include "tpu_mlir/Support/TPUCompressUtil.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

void tpu::MatMulOp::codegen_global_cv18xx(void *ctx, int64_t layer_id) {
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  int64_t batch, M, K, N, right_zp;
  bool with_bias, relu;
  double relu_limit;
  parseParam(batch, M, K, N, with_bias, relu, relu_limit, right_zp);
  assert(batch == 1);
  auto op = getOperation();
  gaddr_t ga_input = Module::getAddress(input());
  gaddr_t ga_filter = Module::getAddress(right());
  gaddr_t ga_output = Module::getAddress(output());
  gaddr_t ga_bias = GA_INVALID;
  bool is_fc = isa<top::WeightOp>(right().getDefiningOp());
  if (is_fc) {
    if (with_bias) {
      ga_bias = Module::getAddress(bias());
    }
    auto multiplier_v = Module::getI64Array(multipliers(), batch, 1);
    auto rshift_v = Module::getI64Array(rshifts(), batch, 0);
    std::vector<int32_t> multiplier_int32;
    std::vector<int32_t> rshift_int32;
    multiplier_int32.assign(multiplier_v->begin(), multiplier_v->end());
    rshift_int32.assign(rshift_v->begin(), rshift_v->end());

    WeightCompresser weight_opt(this->getOperation(), true);
    cvi_backend_tg_fixed_fc_kernel(*backend_ctx, layer_id, ga_input, ga_filter,
                                   ga_bias, ga_output, M, K, N, with_bias, relu,
                                   rshift_int32, multiplier_int32,
                                   &weight_opt.old_data, &weight_opt.new_data,
                                   1, batch, false, false, false);
  } else {
    auto multiplier_v = Module::getI64Array(multipliers(), 1, 1);
    auto rshift_v = Module::getI64Array(rshifts(), 1, 0);
    std::vector<int32_t> multiplier_int32;
    std::vector<int32_t> rshift_int32;
    multiplier_int32.assign(multiplier_v->begin(), multiplier_v->end());
    rshift_int32.assign(rshift_v->begin(), rshift_v->end());

    cvi_backend_tg_fixed_fc_kernel(*backend_ctx, layer_id, ga_input, ga_filter,
                                   ga_bias, ga_output, M, K, N, with_bias, relu,
                                   rshift_int32, multiplier_int32, nullptr, nullptr, 1,
                                   batch, false, false, false);
  }
}

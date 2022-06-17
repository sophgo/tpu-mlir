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
#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fc_global_spec {
  /* common param of float and fixed */
  int32_t R_transpose;
  int32_t have_bias;
  int32_t if_relu;
  float relu_upper_limit;
  /* quantize param */
  int32_t rshift;
  int32_t is_asymmetric;
  int32_t rzp_const_val;
  int32_t rzp_is_const;
  /* requantize param */
  int32_t requant_mode; // mode < 0 means no requantize
  int32_t mul_val;
  int32_t shift_val;
  int32_t offset_val;
} fc_global_spec_t;

#ifdef __cplusplus
}
#endif

void tpu::MatMulOp::codegen_global_int8_bm1684x() {
  int64_t batch, M, K, N;
  bool with_bias, relu;
  parseParam(batch, M, K, N, with_bias, relu);
  assert(batch == 1);
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  fc_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.R_transpose = 0;
  spec.if_relu = relu;
  spec.relu_upper_limit = 0;
  spec.rshift = 0;
  spec.is_asymmetric = 1;
  spec.rzp_is_const = 1;
  spec.rzp_const_val = 0;
  spec.requant_mode = 2;
  spec.mul_val = multiplier();
  spec.shift_val = -rshift();
  auto output_type = Quant::getUniformQuantizedType(output());
  spec.offset_val = output_type.getZeroPoint();
  spec.have_bias = with_bias;
  BM1684x::instance().call_global_func("backend_api_fc_global", &spec,
                                      sizeof(spec), input_spec->data(),
                                      output_spec->data());
}

// f32
void tpu::MatMulOp::codegen_global_float_bm1684x() {

  int64_t batch, M, K, N;
  bool with_bias, relu;
  parseParam(batch, M, K, N, with_bias, relu);
  assert(batch == 1);
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  fc_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.R_transpose = 0;
  spec.if_relu = relu;
  spec.relu_upper_limit = 0;
  spec.have_bias = with_bias;
  BM1684x::instance().call_global_func("backend_api_fc_global", &spec,
                                      sizeof(spec), input_spec->data(),
                                      output_spec->data());
}

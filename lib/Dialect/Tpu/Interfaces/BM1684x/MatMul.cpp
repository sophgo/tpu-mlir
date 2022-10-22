//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
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
  float relu_limit;
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

void tpu::MatMulOp::codegen_global_bm1684x() {
  int64_t batch, M, K, N, right_zp;
  bool with_bias, relu;
  double relu_limit;
  parseParam(batch, M, K, N, with_bias, relu, relu_limit, right_zp);
  assert(batch == 1);
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  BM1684x::fix_shape(input_spec->at(0), {(int32_t)M, (int32_t)K});
  BM1684x::fix_shape(input_spec->at(1), {(int32_t)K, (int32_t)N});
  BM1684x::fix_shape(output_spec->at(0), {(int32_t)M, (int32_t)N});
  fc_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.if_relu = relu;
  spec.relu_limit = relu_limit;
  spec.have_bias = with_bias;
  if (Quant::isUniformQuantized(input())) {
    spec.rshift = 0;
    spec.is_asymmetric = 1;
    spec.rzp_is_const = 1;
    spec.rzp_const_val = right_zp;
    spec.requant_mode = static_cast<int>(quant_mode());
    spec.mul_val = multiplier();
    spec.shift_val = -rshift();
    auto output_type = Quant::getUniformQuantizedType(output());
    spec.offset_val = output_type.getZeroPoint();
  }
  BM1684x::instance().call_global_func("backend_api_fc_global", &spec,
                                       sizeof(spec), input_spec->data(),
                                       output_spec->data());
}

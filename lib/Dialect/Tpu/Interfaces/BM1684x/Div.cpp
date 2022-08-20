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

typedef struct binary_common_spec {
    int32_t binary_type;
    int32_t if_relu;
    float relu_limit;
    int32_t scale_A;
    int32_t scale_B;
    int32_t rshift_A;
    int32_t rshift_B;
} binary_common_spec_t ;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::DivOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  binary_common_spec_t spec;
  memset(&spec, 0, sizeof(binary_common_spec_t));
  spec.binary_type = BM_BINARY_DIV;
  spec.if_relu = (int)do_relu();
  spec.relu_limit = relu_limit().convertToDouble();
  spec.scale_A = 1;
  spec.scale_B = 1;
  spec.rshift_A = 0;
  spec.rshift_B = 0;
  BM1684x::instance().call_global_func("backend_api_eltbinary_global", &spec,
                                       sizeof(spec), input_spec->data(), output_spec->data());
}

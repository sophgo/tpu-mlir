//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::LogOp::codegen_global_int8_bm1684x() {
  auto input_shape = Module::getShape(input());
  active_param_t p = {0};
  p.input_addr = Module::getAddress(input());
  p.output_addr = Module::getAddress(output());
  p.shape_dim = input_shape.size();
  for (int i = 0; i < p.shape_dim; i++) {
    p.shape[i] = input_shape[i];
  }
  p.active_type = ACTIVE_LN;
  p.dtype = BM168x::getDataType(output());
  BM1684x::instance().call_global_func("backend_api_active_global", &p,
                                       sizeof(p));
}

void tpu::LogOp::codegen_global_float_bm1684x() {
  codegen_global_int8_bm1684x();
}

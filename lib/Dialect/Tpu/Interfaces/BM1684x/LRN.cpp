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

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned long long input_addr;
  unsigned long long output_addr;
  int input_n;
  int input_c;
  int input_h;
  int input_w;
  int size;
  float alpha;
  float beta;
  float k;
  int dtype;
} lrn_global_param_t;

#ifdef __cplusplus
}
#endif
// =========================================
// GlobalGenInterface
// =========================================
void tpu::LRNOp::codegen_global_bm1684x() {
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  lrn_global_param_t p = {0};
  p.input_addr = Module::getAddress(input());
  p.output_addr = Module::getAddress(output());
  p.size = size();

  p.input_n = n;
  p.input_c = c;
  p.input_h = h;
  p.input_w = w;

  p.alpha = alpha().convertToDouble();
  p.beta = beta().convertToDouble();
  p.k = bias().convertToDouble();

  p.dtype = BM168x::getDataType(input());
  auto op = getOperation();
  BM168x::call_global_func("backend_api_lrn_global", &p,
                           sizeof(lrn_global_param_t));
}

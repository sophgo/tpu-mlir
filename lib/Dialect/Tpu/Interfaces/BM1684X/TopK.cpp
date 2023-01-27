//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/WeightReorder.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;
using namespace tpu_mlir::bm1684x;

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  int k;
  int dim;
  int descending;
} topk_spec_t;

#ifdef __cplusplus
}
#endif
// =========================================
// GlobalGenInterface
// =========================================
void tpu::TopKOp::codegen_global_bm1684x() {
  // auto op = getOperation();
  // auto input_spec = BM168x::get_input_spec(op);
  // auto output_spec = BM168x::get_output_spec(op);
  // topk_spec_t spec = {0};
  // spec.k = getK();
  // spec.dim = getAxis();
  // spec.descending = getLargest();
  // BM168x::call_global_func("backend_api_topk_global", &spec, sizeof(spec),
  //                          input_spec->data(), output_spec->data());
  llvm_unreachable("Only support dynamic codegen");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::TopKOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) {
    return sizeof(topk_spec_t);
  }
  topk_spec_t spec = {0};
  spec.k = getK();
  spec.dim = getAxis();
  spec.descending = getLargest();
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

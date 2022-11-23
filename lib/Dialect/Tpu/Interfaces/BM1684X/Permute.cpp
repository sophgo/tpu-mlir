//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tranpose_spec {
    uint64_t buffer_global_addr;
    uint32_t order[MAX_SHAPE_DIMS];
    uint32_t is_dynamic;
} transpose_spec_t;

typedef struct transpose_param {
    transpose_spec_t spec;

    int32_t if_getting_buffer_size;
    uint64_t buffer_size_ptr;
} transpose_param_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================
void tpu::PermuteOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  transpose_param_t param = {0};
  param.if_getting_buffer_size = 0;

  auto perm = Module::getI64Array(order());
  auto in_shape = Module::getShape(input());
  int dims = in_shape.size();
  param.spec.buffer_global_addr = 0;
  for (int i = 0; i < dims; i++) {
    param.spec.order[i] =perm->at(i);
  }
  param.buffer_size_ptr = 0;
  BM168x::call_global_func("backend_api_transpose_global", &param,
                                       sizeof(param), input_spec->data(), output_spec->data());
}

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OpDefinition.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include <strings.h>



using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct index_select_common_spec {
  int axis;
  int index_is_coeff; // use for dyn
} index_select_common_spec_t;

typedef struct index_select_global_spec {
  index_select_common_spec_t common;
} index_select_global_spec_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

void tpu::GatherOp::codegen_global_bm1684x() {
  auto op = getOperation();
  index_select_global_spec_t param{0};
  param.common.axis = axis();
  param.common.index_is_coeff = false;
  // assert(module::getStorageType(indices()).isInteger(32));
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_index_select_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

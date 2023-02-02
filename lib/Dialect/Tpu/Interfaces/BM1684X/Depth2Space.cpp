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
#include "tpu_mlir/Support/Module.h"



using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

void tpu::Depth2SpaceOp::codegen_global_bm1684x() {
  depth2space_global_spec_t spec = {0};
  spec.common.block_sizes[0] = getBlockH();
  spec.common.block_sizes[1] = getBlockW();
  spec.common.in_is_nchw = getInIs_NCHW();
  spec.common.out_is_nchw = getOutIs_NCHW();
  spec.common.is_inversed = getIsInversed();
  spec.common.is_crd_mode = getIs_CRD();
  spec.common.swap_cr = getSwapCr();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_depth2space_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::Depth2SpaceOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

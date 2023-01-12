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

typedef struct {
  uint64_t input_global_mem_addr;
  uint64_t output_global_mem_addr;
  int input_shape[MAX_SHAPE_DIMS];
  int dims;
  int block_sizes[2];
  int in_is_nchw;
  int out_is_nchw;
  int is_inversed;
  int is_crd_mode;
  int dtype;
} depth2space_param_t;

#ifdef __cplusplus
}
#endif

void tpu::Depth2SpaceOp::codegen_global_bm1684x() {
  depth2space_param_t param = {0};
  param.input_global_mem_addr = module::getAddress(getInput());
  param.output_global_mem_addr = module::getAddress(getOutput());
  param.dtype = BM168x::getDataType(getOutput());
  auto in_shape = module::getShape(getInput());
  param.dims = in_shape.size();
  for (int i = 0; i < param.dims; i++) {
    param.input_shape[i] = in_shape[i];
  }
  param.block_sizes[0] = getBlockH();
  param.block_sizes[1] = getBlockW();
  param.in_is_nchw = 1;
  param.is_inversed = getIsInversed();
  param.out_is_nchw = 1;
  param.is_crd_mode = getIs_CRD();
  BM168x::call_global_func("backend_api_depth2space_global", &param,
                           sizeof(param));
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::Depth2SpaceOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

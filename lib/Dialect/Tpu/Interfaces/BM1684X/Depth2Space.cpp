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
  param.input_global_mem_addr = module::getAddress(input());
  param.output_global_mem_addr = module::getAddress(output());
  param.dtype = BM168x::getDataType(output());
  auto in_shape = module::getShape(input());
  param.dims = in_shape.size();
  for (int i = 0; i < param.dims; i++) {
    param.input_shape[i] = in_shape[i];
  }
  param.block_sizes[0] = block_h();
  param.block_sizes[1] = block_w();
  param.in_is_nchw = 1;
  param.is_inversed = is_inversed();
  param.out_is_nchw = 1;
  param.is_crd_mode = is_CRD();
  BM168x::call_global_func("backend_api_depth2space_global", &param,
                           sizeof(param));
}

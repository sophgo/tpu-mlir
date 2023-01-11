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
  unsigned long long input_addr;
  unsigned long long *output_addr;
  int shape[MAX_SHAPE_DIMS];
  int shape_dim;
  int split_axis;
  int split_num;
  int *split_size;
  int dtype;
} split_global_param_t;

typedef struct {
  unsigned int input_addr;
  unsigned int *output_addr;
  int shape[4];
  int split_axis;
  int split_num;
  int *split_size;
  int dtype;
} split_local_param_t;

#ifdef __cplusplus
}
#endif

void tpu::SplitOp::codegen_global_bm1684x() {
  std::vector<int64_t> input_shape = module::getShape(getInput());
  split_global_param_t param = {0};
  param.input_addr = module::getAddress(getInput());
  std::vector<unsigned long long> output_addr;
  for (int i = 0; i < getNum(); ++i) {
    output_addr.push_back(module::getAddress(getOutputs()[i]));
  }
  param.output_addr = output_addr.data();
  param.shape_dim = input_shape.size();
  param.split_axis = getAxis();
  param.split_num = getNum();
  for (int i = 0; i < param.shape_dim; ++i) {
    param.shape[i] = input_shape[i];
  }
  int max_split_num = (input_shape[getAxis()] + getNum() - 1) / getNum();
  std::vector<int> split_size;
  for (int i = 0; i < getNum() - 1; ++i) {
    split_size.push_back(max_split_num);
  }
  split_size.push_back(input_shape[getAxis()] - max_split_num * (getNum() - 1));
  param.split_size = split_size.data();
  param.dtype = BM168x::getDataType(getInput());
  BM168x::call_global_func("backend_api_split_global", &param, sizeof(param));
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::SplitOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

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

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned long long input_addr;
    unsigned long long* output_addr;
    int shape[MAX_SHAPE_DIMS];
    int shape_dim;
    int split_axis;
    int split_num;
    int* split_size;
    int dtype;
} split_global_param_t;

typedef struct {
    unsigned int input_addr;
    unsigned int* output_addr;
    int shape[4];
    int split_axis;
    int split_num;
    int* split_size;
    int dtype;
} split_local_param_t;

#ifdef __cplusplus
}
#endif

void tpu::SplitOp::codegen_global_bm1684x() {
  std::vector<int64_t> input_shape = Module::getShape(input());
  split_global_param_t param = {0};
  param.input_addr = Module::getAddress(input());
  std::vector<unsigned long long> output_addr;
  for (int i = 0; i < num(); ++i) {
    output_addr.push_back(Module::getAddress(outputs()[i]));
  }
  param.output_addr = output_addr.data();
  param.shape_dim = input_shape.size();
  param.split_axis = axis();
  param.split_num = num();
  for (int i = 0; i < param.shape_dim; ++i) {
    param.shape[i] = input_shape[i];
  }
  int max_split_num = (input_shape[axis()] + num() - 1) / num();
  std::vector<int> split_size;
  for (int i = 0; i < num() - 1; ++i) {
    split_size.push_back(max_split_num);
  }
  split_size.push_back(input_shape[axis()] - max_split_num * (num() - 1));
  param.split_size = split_size.data();
  param.dtype = BM168x::getDataType(input());
  BM168x::call_global_func("backend_api_split_global", &param,
                                       sizeof(param));
}

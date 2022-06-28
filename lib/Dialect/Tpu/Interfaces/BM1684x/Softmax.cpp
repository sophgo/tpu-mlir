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
  int n;
  int c;
  int h;
  int w;
  bool log;
  float scale_val;
  DATA_TYPE_T dtype;
} softmax_global_param_t;

typedef struct {
  unsigned int input_addr;
  unsigned int output_addr;
  unsigned int buffer_addr;
  int n;
  int c;
  int h;
  int w;
  bool log;
  float scale_val;
  int begin_axis;
  int end_axis;
  int dtype;
} softmax_local_param_t;

#ifdef __cplusplus
}
#endif

void tpu::SoftmaxOp::codegen_global_int8_bm1684x() {
  llvm_unreachable("Codegen to be supported");
}

void tpu::SoftmaxOp::codegen_global_float_bm1684x() {
  softmax_global_param_t param = {0};
  param.input_addr = Module::getAddress(input());
  param.output_addr = Module::getAddress(output());
  int outer_num = 1, softmax_num = 1, inner_num = 1;
  auto in_shape = Module::getShape(input());
  int ax = axis();
  for (uint64_t i = 0; i < ax; i++) {
    outer_num *= in_shape[i];
  }
  softmax_num *= in_shape[ax];
  for (uint64_t i = ax + 1; i < in_shape.size(); i++) {
    inner_num *= in_shape[i];
  }
  param.n = outer_num;
  param.c = softmax_num;
  param.h = 1;
  param.w = inner_num;
  param.log = false;
  param.dtype = BM168x::getDataType(input());
  param.scale_val = 0;
  BM1684x::instance().call_global_func("backend_api_softmax_global", &param,
                                       sizeof(param));
}

//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
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

#ifndef MAX_SHAPE_DIMS
#define MAX_SHAPE_DIMS 8
#endif

typedef struct reduce_full_common_spec {
    int axis[MAX_SHAPE_DIMS];
    int axis_num;
    int method;
    float input_scale;
    float output_scale;
    int keep_dims; // used for dynamic compile
} reduce_full_common_spec_t;

typedef struct reduce_full_global_spec {
    reduce_full_common_spec_t common;
    unsigned long long buffer_addr;
} reduce_full_global_spec_t;

typedef struct reduce_full_global_param {
    reduce_full_global_spec_t spec;
    int if_getting_buffer_size;
    unsigned long long* buffer_size_ptr;
} reduce_full_global_param_t;

#ifdef __cplusplus
}
#endif

void tpu::ReduceOp::codegen_global_bm1684x() {
    auto op = getOperation();
    auto input_spec = BM1684X::get_input_spec(op);
    auto output_spec = BM1684X::get_output_spec(op);
    reduce_full_global_param_t param = {0};
    auto axes_val = Module::getI64Array(axes());
    param.spec.common.axis_num = axes_val->size();
    for (int i = 0; i < param.spec.common.axis_num; i++){
      param.spec.common.axis[i] = axes_val->at(i);
    }
    param.spec.common.method = type();
    param.spec.common.input_scale = 1.0f;
    param.spec.common.output_scale = 1.0f;
    param.spec.common.keep_dims = keepdims();
    param.spec.buffer_addr = 0x0;
    param.if_getting_buffer_size = false;
    BM1684X::instance().call_global_func("backend_api_reduce_full_global", &param,
                                       sizeof(reduce_full_global_param_t), input_spec->data(),
                                       output_spec->data());
}

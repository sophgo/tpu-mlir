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
#include "tpu_mlir/Support/Module.h"




using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MAX_SHAPE_DIMS
#define MAX_SHAPE_DIMS 8
#endif

typedef enum {
  SG_REDUCE_MEAN = 0,
  SG_REDUCE_SUM = 1,
  SG_REDUCE_MAX = 2,
  SG_REDUCE_MIN = 3,
  SG_REDUCE_PROD = 4,
  SG_REDUCE_L2 = 5,
  SG_REDUCE_L1 = 6,
} sg_reduce_method_t;

static int get_reduce_type(llvm::StringRef mode) {
  if (mode == "ReduceMean") {
    return SG_REDUCE_MEAN;
  } else if (mode == "ReduceSum") {
    return SG_REDUCE_SUM;
  } else if (mode == "ReduceMax") {
    return SG_REDUCE_MAX;
  } else if (mode == "ReduceMin") {
    return SG_REDUCE_MIN;
  } else if (mode == "ReduceProd") {
    return SG_REDUCE_PROD;
  } else if (mode == "ReduceL2") {
    return SG_REDUCE_L2;
  } else if (mode == "ReduceL1") {
    return SG_REDUCE_L1;
  } else {
    llvm_unreachable("unsupport reduce mode.");
  }
}

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
  unsigned long long *buffer_size_ptr;
} reduce_full_global_param_t;

#ifdef __cplusplus
}
#endif

void tpu::ReduceOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto attr = parseParam();
  assert(attr.simplified);

  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  std::vector<int32_t> in_shape = {(int)attr.outer_n, (int)attr.outer_c,
                                   (int)attr.axis_dims, (int)attr.inner_dims};
  std::vector<int32_t> out_shape = {(int)attr.outer_n, (int)attr.outer_c, 1,
                                    (int)attr.inner_dims};
  BM168x::fix_shape(input_spec->at(0), in_shape);
  BM168x::fix_shape(output_spec->at(0), out_shape);

  reduce_full_global_param_t param = {0};
  param.spec.common.axis_num = 1;
  param.spec.common.axis[0] = 2;
  param.spec.common.method = get_reduce_type(getMode());
  param.spec.common.input_scale = 1.0f;
  param.spec.common.output_scale = 1.0f;
  param.spec.common.keep_dims = 1;
  param.spec.buffer_addr = module::getAddress(getBuffer());
  param.if_getting_buffer_size = false;
  BM168x::call_global_func("backend_api_reduce_full_global", &param,
                           sizeof(reduce_full_global_param_t),
                           input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ReduceOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

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
#include "tpu_mlir/Support/Module.h"




using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

#ifdef __cplusplus
extern "C" {
#endif

typedef struct strideslice_common_spec {
  int begin_mask;
  int end_mask;
  int begin_index[MAX_SHAPE_DIMS];
  int end_index[MAX_SHAPE_DIMS];
  int strides[MAX_SHAPE_DIMS];
} strideslice_common_spec_t;

typedef struct strideslice_global_spec {
  strideslice_common_spec_t common;
  int shape_size;
  int ellipsis_mask;
  int new_axis_mask;
  int shrink_axis_mask;
  bool is_dynamic;
} strideslice_global_spec_t;

typedef struct strideslice_local_spec {
  strideslice_common_spec_t common;
} strideslice_local_spec_t;

#ifdef __cplusplus
}
#endif

// int8
void tpu::StridedSliceOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  strideslice_common_spec_t param = {0};
  param.begin_mask = begin_mask();
  param.end_mask = end_mask();

  std::vector<int64_t> input_shape = module::getShape(input());
  std::vector<int64_t> output_shape = module::getShape(output());

  auto in_dims = input_shape.size();
  auto out_dims = output_shape.size();
  assert(in_dims == out_dims);
  auto start_v = cast<top::WeightOp>(starts().getDefiningOp()).read<int32_t>();
  auto stride_v = cast<top::WeightOp>(strides().getDefiningOp()).read<int32_t>();
  auto end_v = cast<top::WeightOp>(ends().getDefiningOp()).read<int32_t>();
  for (int i = 0; i < in_dims; i++) {
    param.begin_index[i] = start_v->at(i);
    param.end_index[i] = end_v->at(i);
    param.strides[i] = stride_v->at(i);
  }
  BM168x::call_global_func("backend_api_strideslice_global", &param,
                                       sizeof(param), input_spec->data(),
                                       output_spec->data());
}


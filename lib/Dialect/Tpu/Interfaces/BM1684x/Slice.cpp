//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

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

void tpu::SliceOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);

  strideslice_common_spec_t param = {0};
  param.begin_mask = 0;
  param.end_mask = 0;

  std::vector<int64_t> input_shape = Module::getShape(input());
  std::vector<int64_t> output_shape = Module::getShape(output());

  auto in_dims = input_shape.size();
  auto out_dims = output_shape.size();
  assert(in_dims == out_dims);

  auto offset_v = Module::getI64Array(offset());
  auto steps_v = Module::getI64Array(steps());
  for (int i = 0; i < in_dims; i++) {
    param.begin_index[i] = offset_v->at(i);
    param.end_index[i] = output_shape[i] * steps_v->at(i) + offset_v->at(i);
    param.strides[i] = steps_v->at(i);
  }
  BM1684x::instance().call_global_func("backend_api_strideslice_global", &param,
                                       sizeof(param), input_spec->data(),
                                       output_spec->data());
}

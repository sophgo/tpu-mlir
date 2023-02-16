//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"

#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/LutFunc.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

LogicalResult tpu::ScatterElementsOp::init(InferenceParameter &p) {
  return success();
}
void tpu::ScatterElementsOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ScatterElementsOp::inference(InferenceParameter &p) {
  const float *input = p.inputs[0];
  const float *indices = p.inputs[1];
  const float *updates = p.inputs[2];
  float *output = p.outputs[0];

  const auto input_shape = module::getShape(getInput());
  const auto indices_shape = module::getShape(getIndices());
  const auto updates_shape = module::getShape(getUpdates());
  const int r = input_shape.size();
  const int _axis = getAxis();
  const int axis = _axis < 0 ? _axis + r : _axis;
  assert(0 <= axis && axis < r);

  for (int i = 0; i < r; ++i) {
    if (i != axis) {
      assert(input_shape[i] == indices_shape[i]);
      assert(input_shape[i] == updates_shape[i]);
    } else {
      assert(indices_shape[i] == updates_shape[i]);
    }
  }

  int64_t outer_dim = 1;
  for (int i = 0; i < axis; ++i) {
    outer_dim *= input_shape[i];
  }
  int64_t inner_dim = 1;
  for (int i = axis + 1; i < r; ++i) {
    inner_dim *= input_shape[i];
  }
  const int64_t s = input_shape[axis];
  const int64_t c = indices_shape[axis];
  const int64_t sstride = s * inner_dim;
  const int64_t cstride = c * inner_dim;
  const int64_t hstride = inner_dim;
  const int64_t all_num_elem = outer_dim * sstride;
  const int64_t upd_num_elem = outer_dim * cstride;
  memcpy(output, input, all_num_elem * sizeof(float));
// #pragma omp parallel for schedule(static, omp_schedule(o_num_elem))
  for (int n = 0; n < upd_num_elem; ++n) {
    const int64_t i = n / sstride;
    const int64_t jk = n % sstride;
    const int64_t j = jk / hstride;
    const int64_t k = jk % hstride;
    const float* indices_ik = indices + i * cstride + k;
    const float* updates_ik = updates + i * cstride + k;
    float* output_ik = output + i * sstride + k;
    const int64_t p = (int64_t)indices_ik[j * hstride];
    assert(-s <= p && p < s);
    output_ik[p * hstride] = updates_ik[j * hstride];
  }

  return success();
}

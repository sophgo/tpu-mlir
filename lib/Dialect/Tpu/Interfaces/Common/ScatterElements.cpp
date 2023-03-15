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
      assert(input_shape[i] >= indices_shape[i]);
      assert(input_shape[i] >= updates_shape[i]);
    }
    assert(indices_shape[i] == updates_shape[i]);
  }

  auto all_num_elem = module::getNumElements(getInput());
  auto upd_num_elem = module::getNumElements(getUpdates());
  memcpy(output, input, all_num_elem * sizeof(float));
  const int64_t s = input_shape[axis];

  std::vector<int64_t> in_stride;
  get_stride(input_shape, in_stride);
#pragma omp parallel for schedule(static, omp_schedule(upd_num_elem))
  for (int n = 0; n < upd_num_elem; ++n) {
    std::vector<int64_t> list_(r);
    idx_to_list(n, updates_shape, list_);
    const int64_t p = (int64_t)indices[n];
    assert(-s <= p && p < s);
    list_[axis] = p;
    int64_t in_idx = list_to_idx(list_, in_stride);
    output[in_idx] = updates[n];
  }

  return success();
}

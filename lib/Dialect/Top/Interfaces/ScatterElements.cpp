//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ScatterElementsOp::getFLOPs() { return 0; }

LogicalResult top::ScatterElementsOp::init(InferenceParameter &p) {
  return success();
}
void top::ScatterElementsOp::deinit(InferenceParameter &p) {}

LogicalResult top::ScatterElementsOp::inference(InferenceParameter &p) {
  const float *input = p.inputs[0];
  const float *indices = p.inputs[1];
  const float *updates = p.inputs[2];
  float *output = p.outputs[0];

  const auto input_shape = module::getShape(getInput());
  const auto indices_shape = module::getShape(getIndices());
  const auto updates_shape = module::getShape(getUpdates());
  const int r = input_shape.size();
  const int _axis = getAxis();
  const int replace_add = getReduction();
  const int axis = _axis < 0 ? _axis + r : _axis;
  ASSERT_THIS(0 <= axis && axis < r);

  for (int i = 0; i < r; ++i) {
    if (i != axis) {
      ASSERT_THIS(input_shape[i] >= indices_shape[i]);
      ASSERT_THIS(input_shape[i] >= updates_shape[i]);
    } else {
      ASSERT_THIS(indices_shape[i] == updates_shape[i]);
    }
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
    ASSERT_THIS(-s <= p && p < s);
    list_[axis] = p;
    int64_t in_idx = list_to_idx(list_, in_stride);
    if (replace_add) {
      output[in_idx] += updates[n];
    } else {
      output[in_idx] = updates[n];
    }
  }

  return success();
}

void top::ScatterElementsOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  module::setShapeOrVerify(getOutput(), in_shape);
  if (module::isShape(getInput())) {
    std::vector<std::vector<int64_t>> input_shapes_v;
    for (const auto &input : getOperands()) {
      if (module::isShape(input)) {
        input_shapes_v.push_back(module::getShapeTensorValue(getInput()));
      }
    }
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), input_shapes_v, in_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}

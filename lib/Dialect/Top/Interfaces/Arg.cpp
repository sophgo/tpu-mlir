//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ArgOp::getFLOPs() { return module::getNumElements(getInput()); }

LogicalResult top::ArgOp::init(InferenceParameter &p) { return success(); }
void top::ArgOp::deinit(InferenceParameter &p) {}

typedef enum {
  ARG_MAX,
  ARG_MIN,
} arg_mode_t;

template <typename T>
std::function<bool(T, T)> get_compare_op(arg_mode_t mode, bool select_last) {
  if (select_last) {
    if (mode == ARG_MAX) {
      return std::greater_equal<T>();
    } else {
      return std::less_equal<T>();
    }
  } else {
    if (mode == ARG_MAX) {
      return std::greater<T>();
    } else {
      return std::less<T>();
    }
  }
}

LogicalResult top::ArgOp::inference(InferenceParameter &p) {
  const float *input_v = p.inputs[0];
  float *output_idx = p.outputs[0];
  const bool need_val = !module::isNone(getValues());
  float *output_val = need_val ? p.outputs[1] : nullptr;
  const auto type_val = getMode().str();
  ASSERT_THIS(type_val == "ArgMax" || type_val == "ArgMin");
  const arg_mode_t mode = (type_val == "ArgMax") ? ARG_MAX : ARG_MIN;
  int axis = getAxis();
  auto input_shape = module::getShape(getInput());
  const int input_dims = input_shape.size();
  if (axis < 0) {
    axis += input_dims;
    setAxis(axis);
  }
  ASSERT_THIS(0 <= axis && axis < input_dims);
  int outer_dims =
      std::accumulate(input_shape.begin(), input_shape.begin() + axis, 1,
                      std::multiplies<int64_t>());
  int inner_dims =
      std::accumulate(input_shape.begin() + axis + 1, input_shape.end(), 1,
                      std::multiplies<int64_t>());
  int axis_dims = input_shape[axis];
  const auto cmp_op = get_compare_op<float>(mode, getSelectLastIndex());
  int num_iter = outer_dims * inner_dims;
#pragma omp parallel for schedule(static, omp_schedule(num_iter))
  for (int n = 0; n < num_iter; n++) {
    const int o = n / inner_dims;
    const int i = n % inner_dims;
    const float *input_v_n = input_v + o * axis_dims * inner_dims + i;
    int target_idx = 0;
    float target_val = input_v_n[0];
    for (int a = 1; a < axis_dims; a++) {
      const auto v = input_v_n[a * inner_dims];
      if (cmp_op(v, target_val)) {
        target_val = v;
        target_idx = a;
      }
    }
    output_idx[n] = target_idx;
    if (need_val) {
      output_val[n] = target_val;
    }
  }
  std::vector<int64_t> output_shape;
  output_shape.reserve(input_dims);
  output_shape.assign(input_shape.begin(), input_shape.begin() + axis);
  if (getKeepdims()) {
    output_shape.push_back(1);
  }
  output_shape.insert(output_shape.end(), input_shape.begin() + axis + 1,
                      input_shape.end());
  module::setShape(getIndices(), output_shape);
  if (!module::isNone(getValues())) {
    module::setShape(getValues(), output_shape);
  }

  return success();
}

void top::ArgOp::shape_inference() {
  int64_t axis = getAxis();
  auto input_shape = module::getShape(getInput());
  const int input_dims = input_shape.size();
  if (axis < 0) {
    axis += input_dims;
    setAxis(axis);
  }
  std::vector<int64_t> output_shape;
  output_shape.reserve(input_dims);
  output_shape.assign(input_shape.begin(), input_shape.begin() + axis);
  if (getKeepdims()) {
    output_shape.push_back(1);
  }
  output_shape.insert(output_shape.end(), input_shape.begin() + axis + 1,
                      input_shape.end());
  module::setShapeOrVerify(getIndices(), output_shape);
  if (!module::isNone(getValues())) {
    module::setShapeOrVerify(getValues(), output_shape);
  }
}

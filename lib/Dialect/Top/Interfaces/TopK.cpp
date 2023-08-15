//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::TopKOp::getFLOPs() { return 0; }

LogicalResult top::TopKOp::init(InferenceParameter &p) { return success(); }
void top::TopKOp::deinit(InferenceParameter &p) {}

LogicalResult top::TopKOp::inference(InferenceParameter &p) {
  auto axis = getAxis();
  auto is_largest = getLargest();
  auto K = getKT() ? (int64_t)p.inputs[1][0] : getK();
  auto is_sorted = getSorted();
  if (is_sorted == false) {
    llvm_unreachable("Not supported");
  }
  auto input_shape = module::getShape(getInput());
  if (axis != input_shape.size() - 1) {
    llvm_unreachable("Not supported");
  }
  bool has_values = !module::isNone(getValues());
  bool has_indices = !module::isNone(getIndices());
  int axis_dim = input_shape[axis];
  int outer_dim = 1;
  for (int i = 0; i < axis; i++) {
    outer_dim *= input_shape[i];
  }
#pragma omp parallel for schedule(static, omp_schedule(outer_dim))
  for (int i = 0; i < outer_dim; i++) {
    auto *ptr = p.inputs[0] + i * axis_dim;
    std::vector<std::pair<int, float>> result;
    topk_indices(result, ptr, axis_dim, K, is_largest);
    for (int k = 0; k < K; k++) {
      if (has_indices) {
        auto indices_ptr = p.outputs[1] + i * K + k;
        *indices_ptr = (float)result[k].first;
      }
      if (has_values) {
        auto values_ptr = p.outputs[0] + i * K + k;
        *values_ptr = result[k].second;
      }
    }
  }

  return success();
}

void top::TopKOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  int64_t K = -1;
  if (module::isShape(getKT())) {
    auto kt_vec = module::getShapeTensorValue(getKT());
    assert(kt_vec.size() == 1);
    K = kt_vec[0];
  } else {
    K = getK();
  }
  int64_t axis = getAxis();
  int64_t rank = input_shape.size();
  axis = axis < 0 ? axis + rank : axis;
  setAxis(axis);
  std::vector<int64_t> output_shape(input_shape.size());
  for (int i =0; i < input_shape.size(); i++) {
    if (i == axis) {
      output_shape[i] = K;
    } else {
      output_shape[i] = input_shape[i];
    }
  }
  module::setShapeOrVerify(getValues(), output_shape);
  module::setShapeOrVerify(getIndices(), output_shape);
}



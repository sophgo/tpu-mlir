//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::NonZeroOp::getFLOPs() {
  const auto shape = module::getShape(getInput());
  const auto dims = shape.size();
  const auto num_elem = module::getNumElements(getInput());
  // calc nonzero indices
  const auto flops1 = num_elem;
  // indices -> positions
  const auto flops2 = dims > 1 ? num_elem * dims : 0;
  return flops1 + flops2;
}

LogicalResult top::NonZeroOp::init(InferenceParameter &p) { return success(); }
void top::NonZeroOp::deinit(InferenceParameter &p) {}

static inline int mix(int i, int j, int pos_num, int dims, int order) {
  return order == 0 ? i * dims + j : j * pos_num + i;
}

LogicalResult top::NonZeroOp::inference(InferenceParameter &p) {
  const float *input = p.inputs[0];
  float *output = p.outputs[0];
  const int order = getOrder().str() == "ColMajor" ? 0 : 1;
  const auto num_elem = module::getNumElements(getInput());
  const auto shape = module::getShape(getInput());
  const auto dims = shape.size();
  ASSERT_THIS(dims > 0);
  std::vector<int> indices_;
  indices_.reserve(num_elem);
  // --- step1 : calc nonzero indices ---
  for (int i = 0; i < num_elem; ++i) {
    if (input[i] != 0) {
      indices_.push_back(i);
    }
  }
  const int pos_num = (int)indices_.size();
  // --- step2 : indices -> positions ---
  if (dims > 1) {
#pragma omp parallel for schedule(static, omp_schedule(pos_num))
    for (int i = 0; i < pos_num; ++i) {
      int left = indices_[i];
      for (int j = dims - 1; j >= 0; --j) {
        const int k = mix(i, j, pos_num, dims, order);
        if (shape[j] == 1) {
          output[k] = 0;
        } else {
          output[k] = left % shape[j];
          left /= shape[j];
        }
      }
    }
  } else {
#pragma omp parallel for schedule(static, omp_schedule(pos_num))
    for (int i = 0; i < pos_num; ++i) {
      output[i] = indices_[i];
    }
  }
  std::vector<int64_t> output_shape;
  if (order) {
    output_shape.push_back(dims);
    output_shape.push_back(pos_num);
  } else {
    output_shape.push_back(pos_num);
    output_shape.push_back(dims);
  }
  module::setShape(getOutput(), output_shape);
  return success();
}

void top::NonZeroOp::shape_inference() {
  const int order = getOrder().str() == "ColMajor" ? 0 : 1;
  int64_t num_elem = module::getNumElements(getInput());
  const auto shape = module::getShape(getInput());
  int64_t dims = shape.size();
  std::vector<int64_t> output_shape;
  if (order) {
    output_shape.push_back(dims);
    output_shape.push_back(num_elem);
  } else {
    output_shape.push_back(num_elem);
    output_shape.push_back(dims);
  }
  module::setShapeOrVerify(getOutput(), output_shape);
  // set top run mode to dynamic
  module::setTopRunMode(module::TopRunMode::DYNAMIC);
}

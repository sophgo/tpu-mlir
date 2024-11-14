//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::IndexPutOp::getFLOPs() { return 0; }

LogicalResult top::IndexPutOp::init(InferenceParameter &p) { return success(); }
void top::IndexPutOp::deinit(InferenceParameter &p) {}

LogicalResult top::IndexPutOp::inference(InferenceParameter &p) {
  const float *src = p.inputs[0];
  const float *inds = p.inputs[1];
  const float *vals = p.inputs[2];
  float *dst = p.outputs[0];
  auto input_shape = module::getShape(getInput());
  auto index_shape = module::getShape(getIndices());
  ASSERT_THIS(index_shape.size() == 1);
  int64_t input_num = module::getNumElements(getInput());
  int64_t num_indices = module::getNumElements(getIndices());

  int64_t inner_dims = std::accumulate(
      input_shape.begin() + 1, input_shape.end(), 1, std::multiplies<int>());
  std::memcpy(dst, src, input_num * sizeof(float));
#pragma omp parallel for schedule(static, omp_schedule(num_indices))
  for (int64_t i = 0; i < num_indices; ++i) {
    for (int64_t j = 0; j < inner_dims; ++j) {
      int64_t dst_idx = (inds[i] * inner_dims) + j;
      int64_t val_idx = (i * inner_dims) + j;
      if (getAccumulate()) {
        dst[dst_idx] += vals[val_idx];
      } else
        dst[dst_idx] = vals[val_idx];
    }
  }
  return success();

  //   auto num_indices = module::getNumElements(getIndices());
  //   auto ax = getAxis();
  //   int64_t outer_dims = 1;
  //   int64_t inner_dims = 1;
  //   auto input_shape = module::getShape(getInput());
  //   if (ax < 0) {
  //     ax += input_shape.size();
  //   }
  //   for (int i = 0; i < ax; ++i) {
  //     outer_dims *= input_shape[i];
  //   }
  //   for (int i = ax + 1; i < input_shape.size(); ++i) {
  //     inner_dims *= input_shape[i];
  //   }

  //   auto num_elems = module::getNumElements(getOutput());
  // #pragma omp parallel for schedule(static, omp_schedule(num_elems))
  //   for (int64_t i = 0; i < outer_dims; ++i) {
  //     for (int64_t j = 0; j < num_indices; ++j) {
  //       for (int64_t k = 0; k < inner_dims; ++k) {
  //         int64_t src_idx =
  //             (i * input_shape[ax] +
  //              (int64_t)(inds[j] < 0 ? inds[j] + input_shape[ax] : inds[j]))
  //              *
  //                 inner_dims +
  //             k;
  //         int64_t dst_idx = (i * num_indices + j) * inner_dims + k;
  //         dst[dst_idx] = src[src_idx];
  //       }
  //     }
  //   }

  // return success();
}

void top::IndexPutOp::shape_inference() {
  common_shape_inference(getOperation());
}

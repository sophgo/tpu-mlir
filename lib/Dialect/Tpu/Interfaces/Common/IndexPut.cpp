//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::IndexPutOp::init(InferenceParameter &p) { return success(); }
void tpu::IndexPutOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::IndexPutOp::inference(InferenceParameter &p) {
  const float *src = p.inputs[0];
  const float *inds = p.inputs[1];
  const float *vals = p.inputs[2];
  float *dst = p.outputs[0];
  auto input_shape = module::getShape(getInput());
  int64_t input_num = module::getNumElements(getInput());
  int64_t num_indices = module::getNumElements(getIndices());

  int64_t inner_dims = std::accumulate(input_shape.begin()+1, input_shape.end(), 1, std::multiplies<int>());
  std::memcpy(dst, src, input_num * sizeof(float));
#pragma omp parallel for schedule(static, omp_schedule(num_indices))
  for(int64_t i = 0; i < num_indices; ++i){
    for(int64_t j = 0; j < inner_dims; ++j){
      int64_t dst_idx = (inds[i] * inner_dims) + j;
      int64_t val_idx = (i * inner_dims) + j;
      if(getAccumulate()){
        dst[dst_idx] += vals[val_idx];}
      else
        dst[dst_idx] = vals[val_idx];
    }
  }
  return success();
}


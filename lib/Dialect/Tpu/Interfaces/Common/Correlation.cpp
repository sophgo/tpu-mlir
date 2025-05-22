//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::CorrelationOp::init(InferenceParameter &p) {
  return success();
}
void tpu::CorrelationOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::CorrelationOp::inference(InferenceParameter &p) {

  auto lhs_shape = module::getShape(getInputs()[0]);
  if (lhs_shape.size() != 4) {
    return failure();
  }
  int64_t max_disp = getMaxDisp();
  int64_t num_groups = getNumGroups();
  ASSERT_THIS(lhs_shape[1] % num_groups == 0);

  float *lsrc = p.inputs[0];
  float *rsrc = p.inputs[1];
  float *dst = p.outputs[0];

  int64_t in = num_groups;
  int64_t ic = lhs_shape[1] / num_groups;
  int64_t ih = lhs_shape[2];
  int64_t iw = lhs_shape[3];

  const int64_t spatial_dim = ih * iw;

#pragma omp parallel for collapse(2) schedule(static)
  for (int64_t group = 0; group < in; ++group) {
    for (int64_t cut_idx = 0; cut_idx < max_disp; ++cut_idx) {
      const float *lsrc_group = lsrc + group * ic * spatial_dim;
      const float *rsrc_group = rsrc + group * ic * spatial_dim;
      float *dst_group = dst + group * max_disp * spatial_dim;

      float *output = dst_group + cut_idx * spatial_dim;
      int64_t w_start = cut_idx;

      for (int64_t h_idx = 0; h_idx < ih; ++h_idx) {
        const int64_t hw = h_idx * iw;
        for (int64_t w_idx = w_start; w_idx < iw; ++w_idx) {
          const int64_t wcut = w_idx - cut_idx;
          float sum = 0.0f;
          for (int64_t c = 0; c < ic; ++c) {
            const int64_t c_offset = c * spatial_dim;
            sum += lsrc_group[c_offset + hw + w_idx] *
                   rsrc_group[c_offset + hw + wcut];
          }
          output[hw + w_idx] = sum / ic;
        }
        if (cut_idx > 0) {
          for (int64_t w_idx = 0; w_idx < w_start; ++w_idx) {
            output[hw + w_idx] = 0.0f;
          }
        }
      }
    }
  }
  return success();
}

bool tpu::CorrelationOp::support_multi_core() { return false; }

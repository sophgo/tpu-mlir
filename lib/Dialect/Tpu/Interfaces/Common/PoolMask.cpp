//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::PoolMaskOp::init(InferenceParameter &p) { return success(); }
void tpu::PoolMaskOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::PoolMaskOp::inference(InferenceParameter &p) {
  auto input_shape = module::getShape(getInput());
  auto output_shape = module::getShape(getOutput());
  int64_t _scale = getScale();
  int n = input_shape[0];
  int c = input_shape[1];
  int h = input_shape[2];
  int w = input_shape[3];
  int h_ex = output_shape[2];
  int w_ex = output_shape[3];
#pragma omp parallel for schedule(static, omp_schedule(n *c))
  for (int n_idx = 0; n_idx < n * c; n_idx++) {
    for (int h_idx = 0; h_idx < h_ex; h_idx += _scale) {
      for (int w_idx = 0; w_idx < w_ex; w_idx += _scale) {
        int index = n_idx * h * w + h_idx * w + w_idx;
        float max = p.inputs[0][index];
        int out_index = n_idx * h_ex * w_ex + h_idx * w_ex + w_idx;
        int max_index = out_index;
        for (int pool_h = 0; pool_h < _scale && (pool_h + h_idx < h);
             pool_h++) {
          for (int pool_w = 0; pool_w < _scale && (pool_w + w_idx < w);
               pool_w++) {
            int pool_index = index + pool_h * w + pool_w;
            if (p.inputs[0][pool_index] > max) {
              max = p.inputs[0][pool_index];
              max_index = out_index + pool_h * w_ex + pool_w;
            }
          }
        }
        p.outputs[0][max_index] = 1.0f;
      }
    }
  }
  return success();
}

bool tpu::PoolMaskOp::support_multi_core() { return false; }

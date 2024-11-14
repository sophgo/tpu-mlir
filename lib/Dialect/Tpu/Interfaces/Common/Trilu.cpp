//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

LogicalResult tpu::TriluOp::init(InferenceParameter &p) { return success(); }
void tpu::TriluOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::TriluOp::inference(InferenceParameter &p) {
  auto upper = getUpper();
  auto diagonal = getDiagonal();
  auto in_shape = module::getShape(getInput());
  float *in = p.inputs[0];
  float *out = p.outputs[0];
  auto dims = in_shape.size();
  int64_t H = in_shape[dims - 2];
  int64_t W = in_shape[dims - 1];
  int64_t N = 1;

#pragma omp parallel for schedule(static, omp_schedule(dims - 2))
  for (int64_t i = 0; i < dims - 2; ++i) {
    N *= in_shape[i];
  }
#pragma omp parallel for schedule(static, omp_schedule(N))
  for (int64_t nidx = 0; nidx < N; ++nidx) {
    float *in_n = in + nidx * W * H;
    float *out_n = out + nidx * W * H;
#pragma omp parallel for schedule(static, omp_schedule(H))
    for (int64_t y = 0; y < H; ++y) {
#pragma omp parallel for schedule(static, omp_schedule(W))
      for (int64_t x = 0; x < W; ++x) {
        out_n[y * W + x] = (upper ? (x - y >= diagonal) : (x - y <= diagonal))
                               ? in_n[y * W + x]
                               : 0;
      }
    }
  }
  return success();
}

bool tpu::TriluOp::support_multi_core() { return false; }

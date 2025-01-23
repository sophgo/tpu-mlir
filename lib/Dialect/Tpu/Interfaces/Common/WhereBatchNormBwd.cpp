//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::WhereBnbwdOp::init(InferenceParameter &p) {

  return success();
}

void tpu::WhereBnbwdOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::WhereBnbwdOp::inference(InferenceParameter &p) {
//   const auto input_shape = module::getShape(getInput());
//   const int N = input_shape[0];
//   const int C = input_shape[1];
//   const int H = input_shape.size() > 2 ? input_shape[2] : 1;
//   const int W = input_shape.size() > 3 ? input_shape[3] : 1;
//   const float M = N * H * W;
//   const float *dout = p.inputs[0];
//   const float *x_ = p.inputs[1];
//   const float *gamma = p.inputs[2];
//   const float *mean = p.inputs[3];
//   const float *var = p.inputs[4];

//   float *dx = p.outputs[0];
//   float *dgamma = p.outputs[1];
//   float *dbeta = p.outputs[2];

//   std::vector<float> dxhut(C * M, 0);
//   std::vector<float> dx2_tmp(C, 0);
//   std::vector<float> dx2(C, 0);
//   std::vector<float> dx3(C, 0);

// #pragma omp parallel for schedule(static, omp_schedule(C))
//   for (int c = 0; c < C; ++c) {
//     float rstd = var[c];
//     for (int n = 0; n < N; ++n) {
//       for (int h = 0; h < H; ++h) {
//         for (int w = 0; w < W; ++w) {
//           int idx = ((n * C + c) * H + h) * W + w;
//           float x_hat = (x_[idx] - mean[c]) * rstd;
//           dgamma[c] += dout[idx] * x_hat;
//           dbeta[c] += dout[idx];
//           dxhut[idx] = dout[idx] * gamma[c];
//           dx2[c] += std::pow(rstd, 2) * dxhut[idx] * (x_[idx] - mean[c]);
//           dx3[c] += dxhut[idx];
//         }
//       }
//     }
//   }

// #pragma omp parallel for schedule(static, omp_schedule(C))
//   for (int c = 0; c < C; ++c) {
//     float rstd = var[c];
//     for (int n = 0; n < N; ++n) {
//       for (int h = 0; h < H; ++h) {
//         for (int w = 0; w < W; ++w) {
//           int idx = ((n * C + c) * H + h) * W + w;
//           dx[idx] = (rstd / M) *
//                     (M * dxhut[idx] - dx2[c] * (x_[idx] - mean[c]) - dx3[c]);
//         }
//       }
//     }
//   }

  return success();
}

bool tpu::WhereBnbwdOp::support_multi_core() { return true; }


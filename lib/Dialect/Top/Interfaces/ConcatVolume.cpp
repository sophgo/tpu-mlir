//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ConcatVolumeOp::getFLOPs() { return 0; }

LogicalResult top::ConcatVolumeOp::init(InferenceParameter &p) {
  return success();
}

void top::ConcatVolumeOp::deinit(InferenceParameter &p) {}

LogicalResult top::ConcatVolumeOp::inference(InferenceParameter &p) {
  auto lhs_shape = module::getShape(getInputs()[0]);

  if (lhs_shape.size() != 4) {
    return failure();
  }

  int64_t B = lhs_shape[0];
  int64_t C = lhs_shape[1];
  int64_t H = lhs_shape[2];
  int64_t W = lhs_shape[3];
  int64_t max_disp = getMaxDisp();

  float *left = p.inputs[0];
  float *right = p.inputs[1];
  float *output = p.outputs[0];

  // Initialize output to 0
  std::memset(output, 0, B * 2 * C * max_disp * H * W * sizeof(float));

  // Output shape: (B, 2*C, max_disp, H, W)
  // Memory layout: B * (2*C) * max_disp * H * W
  const int64_t out_c_stride = max_disp * H * W;
  const int64_t out_d_stride = H * W;
  const int64_t in_c_stride = H * W;

#pragma omp parallel for collapse(2) schedule(static)
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t d = 0; d < max_disp; ++d) {
      float *out_base = output + b * 2 * C * max_disp * H * W;
      const float *left_base = left + b * C * H * W;
      const float *right_base = right + b * C * H * W;

      // Copy left feature to first C channels
      for (int64_t c = 0; c < C; ++c) {
        for (int64_t h = 0; h < H; ++h) {
          for (int64_t w = 0; w < W; ++w) {
            int64_t in_idx = c * in_c_stride + h * W + w;
            int64_t out_idx = c * out_c_stride + d * out_d_stride + h * W + w;
            out_base[out_idx] = left_base[in_idx];
          }
        }
      }

      // Copy right feature to last C channels (with shift)
      int64_t w_start = d;
      for (int64_t c = 0; c < C; ++c) {
        for (int64_t h = 0; h < H; ++h) {
          for (int64_t w = w_start; w < W; ++w) {
            int64_t in_idx = c * in_c_stride + h * W + (w - d);
            int64_t out_idx =
                (C + c) * out_c_stride + d * out_d_stride + h * W + w;
            out_base[out_idx] = right_base[in_idx];
          }
        }
      }
    }
  }
  return success();
}

void top::ConcatVolumeOp::shape_inference() {
  int64_t max_disp = getMaxDisp();
  ASSERT_THIS(getInputs().size() == 2 && max_disp > 0);
  auto lhs_shape = module::getShape(getInputs()[0]);
  auto rhs_shape = module::getShape(getInputs()[1]);
  ASSERT_THIS(lhs_shape == rhs_shape);
  ASSERT_THIS(lhs_shape.size() == 4);

  // Input: (B, C, H, W)
  // Output: (B, 2*C, max_disp, H, W)
  std::vector<int64_t> out_shape = {
      lhs_shape[0],     // B
      lhs_shape[1] * 2, // 2*C
      max_disp,         // max_disp
      lhs_shape[2],     // H
      lhs_shape[3]      // W
  };
  module::setShapeOrVerify(getOutput(), out_shape);
}

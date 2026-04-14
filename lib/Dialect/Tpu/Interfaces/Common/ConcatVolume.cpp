//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::ConcatVolumeOp::init(InferenceParameter &p) {
  return success();
}

void tpu::ConcatVolumeOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ConcatVolumeOp::inference(InferenceParameter &p) {
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

bool tpu::ConcatVolumeOp::support_multi_core() {
  if (!module::isBM1688() && !module::isBM1690Family()) {
    return false;
  }

  auto input_shape = module::getShape(getInputs()[0]);
  if (input_shape.size() != 4) {
    return false;
  }

  int64_t batch = input_shape[0];
  int64_t C = input_shape[1];

  // Split by batch when batch >= 2
  if (batch >= 2) {
    return true;
  }

  // Split by channel (2*C) when batch == 1
  if (batch == 1 && 2 * C >= 4) {
    return true;
  }

  return false;
}

ArrayAttr tpu::ConcatVolumeOp::getIndexingMaps() {
  MLIRContext *context = getContext();

  auto input_shape = module::getShape(getInputs()[0]);
  if (input_shape.size() != 4) {
    return Builder(getContext()).getAffineMapArrayAttr({});
  }

  int64_t batch = input_shape[0];
  int64_t C = input_shape[1];

  if (batch >= 2) {
    // Split by batch dimension
    AffineMap inputMap = AffineMap::getMultiDimIdentityMap(1, context);
    AffineMap outputMap = AffineMap::getMultiDimIdentityMap(1, context);

    SmallVector<AffineMap> indexingMaps{inputMap, inputMap, outputMap};
    return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);

  } else if (batch == 1 && 2 * C >= 4) {
    // Split by channel dimension (each channel is independent)
    AffineMap inputMap = AffineMap::getMultiDimIdentityMap(2, context);
    AffineMap outputMap = AffineMap::getMultiDimIdentityMap(2, context);

    SmallVector<AffineMap> indexingMaps{inputMap, inputMap, outputMap};
    return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);

  } else {
    return Builder(getContext()).getAffineMapArrayAttr({});
  }
}

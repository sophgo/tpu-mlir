//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/LutFunc.h"

LogicalResult tpu::SelectiveScanOp::init(InferenceParameter &p) {
  return success();
}

void tpu::SelectiveScanOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::SelectiveScanOp::inference(InferenceParameter &p) {

  auto deltaA_shape = module::getShape(getDeltaA());    // [N, Kcdim, L, Batch]
  auto deltaB_u_shape = module::getShape(getDeltaBU()); // [N, Kcdim, L, Batch]
  auto c_shape = module::getShape(getCs());             // [L, Kcdim, 1, Batch]

  if (deltaA_shape.size() != 4 || deltaB_u_shape.size() != 4 ||
      c_shape.size() != 4) {
    return failure();
  }
  if (deltaA_shape != deltaB_u_shape || deltaA_shape[2] != c_shape[0] ||
      deltaA_shape[3] != c_shape[3] || deltaA_shape[1] != c_shape[1]) {
    return failure();
  }

  // int N = deltaA_shape[0];
  int Kcdim = deltaA_shape[1];
  int L = deltaA_shape[2];
  int Batch = deltaA_shape[3];
  int Cdim_plus_2 = Kcdim / 2;

  const float *c_ptr = p.inputs[0];
  const float *deltaA_ptr = p.inputs[1];
  const float *deltaB_u_ptr = p.inputs[2];
  const float *u_ptr = p.inputs[3];
  const float *D_ptr = p.inputs[4];
  float *output_ptr = p.outputs[0]; //  [L, Kcdim, Batch]

  bool has_u = !getUs().getType().isa<NoneType>();
  bool has_D = !getDs().getType().isa<NoneType>();

  std::vector<float> x_up(Cdim_plus_2 * Batch, 0.0f);
  std::vector<float> x_down(Cdim_plus_2 * Batch, 0.0f);

  for (int i = 0; i < L; i++) {
#pragma omp parallel for collapse(2)
    for (int k = 0; k < Cdim_plus_2; k++) {
      for (int b = 0; b < Batch; b++) {
        int state_idx = k * Batch + b;
        int delta_idx = k * (L * Batch) + i * Batch + b;

        // x = deltaA * x + deltaB_u
        x_up[state_idx] =
            deltaA_ptr[delta_idx] * x_up[state_idx] + deltaB_u_ptr[delta_idx];

        // y_up = x * c
        int c_idx = i * (Kcdim * Batch) + k * Batch + b;
        int out_idx = i * (Kcdim * Batch) + k * Batch + b;
        output_ptr[out_idx] = x_up[state_idx] * c_ptr[c_idx];
      }
    }

    int rev_i = L - 1 - i;
#pragma omp parallel for collapse(2)
    for (int k = 0; k < Cdim_plus_2; k++) {
      for (int b = 0; b < Batch; b++) {
        int state_idx = k * Batch + b;
        int delta_idx = (Cdim_plus_2 + k) * (L * Batch) + rev_i * Batch + b;

        x_down[state_idx] =
            deltaA_ptr[delta_idx] * x_down[state_idx] + deltaB_u_ptr[delta_idx];

        // y_down = x * c
        int c_idx = rev_i * (Kcdim * Batch) + (Cdim_plus_2 + k) * Batch + b;
        int out_idx = rev_i * (Kcdim * Batch) + (Cdim_plus_2 + k) * Batch + b;
        output_ptr[out_idx] = x_down[state_idx] * c_ptr[c_idx];
      }
    }
  }

  // y + u * D
  if (has_u && has_D) {
#pragma omp parallel for collapse(3)
    for (int l = 0; l < L; l++) {
      for (int k = 0; k < Kcdim; k++) {
        for (int b = 0; b < Batch; b++) {
          int idx = l * (Kcdim * Batch) + k * Batch + b;
          output_ptr[idx] += u_ptr[idx] * D_ptr[k];
        }
      }
    }
  }

  return success();
}

bool tpu::SelectiveScanOp::support_multi_core() { return false; }

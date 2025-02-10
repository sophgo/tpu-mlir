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
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::BatchNormTrainOp::init(InferenceParameter &p) {

  return success();
}

void tpu::BatchNormTrainOp::deinit(InferenceParameter &p) {}

static void normalize_f32(const float *input_data, float *output_data,
                          std::vector<float> mean, std::vector<float> var,
                          float *running_mean, float *running_var,
                          const float *scale_data, const float *shift_data,
                          const float eps, const float momentum,
                          const int batch_size, const int channels,
                          const int spatial_size, int c) {
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < spatial_size; ++s) {
      const int index = b * channels * spatial_size + c * spatial_size + s;
      mean[c] += input_data[index];
      var[c] += input_data[index] * input_data[index];
    }
  }
  mean[c] /= (batch_size * spatial_size);
  var[c] /= (batch_size * spatial_size);
  var[c] -= mean[c] * mean[c]; // variance = E[X^2] - (E[X])^2
  // TODO checke eps add time
  var[c] += eps;

  // Update running averages
  running_mean[c] = momentum * running_mean[c] + (1 - momentum) * mean[c];
  running_var[c] = momentum * running_var[c] + (1 - momentum) * var[c];

  // Normalize and scale/shift
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < spatial_size; ++s) {
      const int index = b * channels * spatial_size + c * spatial_size + s;
      output_data[index] = (input_data[index] - mean[c]) / std::sqrt(var[c]);
      if (scale_data) {
        output_data[index] *= scale_data[c];
      }
      if (shift_data) {
        output_data[index] += shift_data[c];
      }
    }
  }
}

static void normalize_f16(const float *input_data, float *output_data,
                          std::vector<float> mean, std::vector<float> var,
                          float *running_mean, float *running_var,
                          const float *scale_data, const float *shift_data,
                          const float eps, const float momentum,
                          const int batch_size, const int channels,
                          const int spatial_size, int c) {
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < spatial_size; ++s) {
      const int index = b * channels * spatial_size + c * spatial_size + s;
      mean[c] += F16(input_data[index]);
      var[c] += F16(input_data[index]) * F16(input_data[index]);
    }
  }
  mean[c] = mean[c] * F16(1.0f / (batch_size * spatial_size));
  var[c] = var[c] * F16(1.0f / (batch_size * spatial_size));
  var[c] -= F16(mean[c]) * F16(mean[c]); // variance = E[X^2] - (E[X])^2
  var[c] += F16(eps);

  // Update running averages
  running_mean[c] =
      F16(momentum) * F16(running_mean[c]) + F16((1 - momentum)) * F16(mean[c]);
  running_var[c] =
      F16(momentum) * F16(running_var[c]) + F16((1 - momentum)) * F16(var[c]);

  // Normalize and scale/shift
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < spatial_size; ++s) {
      const int index = b * channels * spatial_size + c * spatial_size + s;
      output_data[index] = F16((F16(input_data[index]) - F16(mean[c])) /
                               F16(std::sqrt(F16(var[c]))));
      if (scale_data) {
        output_data[index] *= F16(scale_data[c]);
      }
      if (shift_data) {
        output_data[index] += F16(shift_data[c]);
      }
    }
  }
}

static void simulate_fp16(float *args, int size) {
  for (int i = 0; i < size; i++) {
    args[i] = (float)F16(args[i]);
  }
}

LogicalResult tpu::BatchNormTrainOp::inference(InferenceParameter &p) {
  const float momentum = getMomentum().convertToDouble();
  const float eps = getEpsilon().convertToDouble();
  const auto input_shape = module::getShape(getInput());
  auto out_type = module::getStorageType(getOutput());
  ;

  int n = input_shape[0];
  int c = input_shape[1];
  int h = input_shape[2];
  int w = input_shape[3];

  const float *input_data = p.inputs[0];
  const float *mean_data = p.inputs[1];
  const float *var_data = p.inputs[2];
  const float *gamma_data = p.inputs[3];
  const float *beta_data = p.inputs[4];

  bool do_relu = getDoRelu();

  float *output_data = p.outputs[0];
  float *saved_mean_data = p.outputs[1];
  float *saved_invstd = p.outputs[2];
  float *running_mean = p.outputs[3];
  float *running_var = p.outputs[4];

  float *var_temp = new float[c];

  for (int ci = 0; ci < c; ci++) {
    float cur_mean = 0;
    for (int ni = 0; ni < n; ni++) {
      for (int hi = 0; hi < h; hi++) {
        for (int wi = 0; wi < w; wi++) {
          int index = ni * c * h * w + ci * h * w + hi * w + wi;
          cur_mean += input_data[index];
        }
      }
    }
    cur_mean /= n * h * w;
    saved_mean_data[ci] = cur_mean;
  }

  for (int ci = 0; ci < c; ci++) {
    float cur_var = 0;
    float cur_save_mean = saved_mean_data[ci];
    for (int ni = 0; ni < n; ni++) {
      for (int hi = 0; hi < h; hi++) {
        for (int wi = 0; wi < w; wi++) {
          int index = ni * c * h * w + ci * h * w + hi * w + wi;
          cur_var += (input_data[index] - cur_save_mean) *
                     (input_data[index] - cur_save_mean);
        }
      }
    }
    cur_var /= n * h * w;
    var_temp[ci] = cur_var;
    saved_invstd[ci] = 1 / std::sqrt(cur_var + eps);
  }

  // running mean var
  for (int ci = 0; ci < c; ci++) {
    running_mean[ci] =
        (1 - momentum) * mean_data[ci] + momentum * saved_mean_data[ci];
    running_var[ci] = (1 - momentum) * var_data[ci] + momentum * var_temp[ci];
  }

  for (int ni = 0; ni < n; ni++) {
    for (int ci = 0; ci < c; ci++) {
      for (int hi = 0; hi < h; hi++) {
        for (int wi = 0; wi < w; wi++) {
          int index = ni * c * h * w + ci * h * w + hi * w + wi;
          output_data[index] =
              (input_data[index] - saved_mean_data[ci]) * saved_invstd[ci];
          output_data[index] *= gamma_data[ci];
          output_data[index] += beta_data[ci];
          if (do_relu) {
            output_data[index] =
                output_data[index] > 0 ? output_data[index] : 0;
          }
        }
      }
    }
  }
  if (out_type.isF16()) {
    // simulate fp16
    simulate_fp16(output_data, n * c * h * w);
    simulate_fp16(saved_mean_data, c);
    simulate_fp16(saved_invstd, c);
    simulate_fp16(running_mean, c);
    simulate_fp16(running_var, c);
  }
  delete[] var_temp;
  return success();
}

uint32_t tpu::BatchNormTrainOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::BatchNormTrainOp::get_fw_type_bm1684() { return -1; }

bool tpu::BatchNormTrainOp::support_multi_core() { return true; }

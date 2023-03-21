//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

int64_t top::GroupNormOp::getFLOPs() {
  const bool have_weight = !module::isNone(getWeight());
  const bool have_bias = !module::isNone(getBias());
  const int64_t num_elem = module::getNumElements(getOutput());
  return num_elem * (10 + have_weight + have_bias);
}

LogicalResult top::GroupNormOp::init(InferenceParameter &p) {
  return success();
}
void top::GroupNormOp::deinit(InferenceParameter &p) {}

LogicalResult top::GroupNormOp::inference(InferenceParameter &p) {
  const float eps_ = getEps().convertToDouble();
  const auto input_shape = module::getShape(getInput());
  const int channel = input_shape[1];
  const int num_groups = getNumGroups();
  assert(channel % num_groups == 0);
  const int channel_per_group = channel / num_groups;

  int outer_dim = input_shape[0] * num_groups;
  int inner_dim = channel_per_group;
  for (int i = 2; i < input_shape.size(); i++) {
    inner_dim *= input_shape[i];
  }

  const bool have_weight = !module::isNone(getWeight());
  const bool have_bias = !module::isNone(getBias());

  const float *input_data = p.inputs[0];
  const float *weight_data = have_weight ? p.inputs[1] : nullptr;
  const float *bias_data = have_bias ? p.inputs[2] : nullptr;
  float *output_data = p.outputs[0];

  std::vector<float> mean_arr(outer_dim, 0);
  std::vector<float> rstd_arr(outer_dim, 0);

#pragma omp parallel for schedule(static, omp_schedule(outer_dim))
  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < inner_dim; ++j) {
      mean_arr[i] += input_data[i * inner_dim + j];
    }
    mean_arr[i] /= inner_dim;
    for (int j = 0; j < inner_dim; ++j) {
      const float dij = input_data[i * inner_dim + j] - mean_arr[i];
      rstd_arr[i] += dij * dij;
    }
    rstd_arr[i] /= inner_dim;
    rstd_arr[i] += eps_;
    rstd_arr[i] = std::sqrt(rstd_arr[i]);
    rstd_arr[i] = 1.0f / rstd_arr[i];
    for (int j = 0; j < inner_dim; ++j) {
      output_data[i * inner_dim + j] =
          input_data[i * inner_dim + j] - mean_arr[i];
      output_data[i * inner_dim + j] *= rstd_arr[i];
    }
  }
  int num_channels = channel * input_shape[0];
  int num_iter = module::getNumElements(getOutput()) / num_channels;
#pragma omp parallel for schedule(static, omp_schedule(num_iter))
  for (int i = 0; i < num_channels; ++i) {
    int c = num_channels % channel;
    float *output_i = output_data + i * num_iter;
    auto weight = have_weight ? weight_data[c] : 1.0;
    auto bias = have_bias ? bias_data[c] : 0.0;
    if (weight == 1.0 && bias == 0.0) {
      // do nothing
    } else {
      for (int j = 0; j < num_iter; ++j) {
        output_i[j] = output_i[j] * weight + bias;
      }
    }
  }
  return success();
}

void top::GroupNormOp::shape_inference() {
  common_shape_inference(getOperation());
}

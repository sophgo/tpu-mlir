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

int64_t top::GroupNormTrainOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult top::GroupNormTrainOp::init(InferenceParameter &p) {
  return success();
}
void top::GroupNormTrainOp::deinit(InferenceParameter &p) {}

LogicalResult top::GroupNormTrainOp::inference(InferenceParameter &p) {
  const float eps_ = getEps().convertToDouble();
  const auto input_shape = module::getShape(getInput());
  const int channel = input_shape[1];
  const int num_groups = getNumGroups();
  ASSERT_THIS(channel % num_groups == 0);
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
  float *mean_arr = p.outputs[1];
  float *rstd_arr = p.outputs[2];

#pragma omp parallel for
  for (int i = 0; i < outer_dim; ++i) {
    mean_arr[i] = 0.0f;
    rstd_arr[i] = 0.0f;
  }
#pragma omp parallel for schedule(static, omp_schedule(outer_dim))
  for (int i = 0; i < outer_dim; ++i) {
    float mean = 0.0f;
    float rstd = 0.0f;
    for (int j = 0; j < inner_dim; ++j) {
      mean += input_data[i * inner_dim + j];
    }
    mean /= inner_dim;
    for (int j = 0; j < inner_dim; ++j) {
      const float dij = input_data[i * inner_dim + j] - mean;
      rstd += dij * dij;
    }
    rstd /= inner_dim;
    rstd += eps_;
    rstd = std::sqrt(rstd);
    rstd = 1.0f / rstd;
    mean_arr[i] = mean;
    rstd_arr[i] = rstd;
    for (int j = 0; j < inner_dim; ++j) {
      output_data[i * inner_dim + j] =
          (input_data[i * inner_dim + j] - mean) * rstd;
    }
  }

  inner_dim /= channel_per_group;
  int num_iter = module::getNumElements(getOutput()) / channel;
#pragma omp parallel for schedule(static, omp_schedule(num_iter))
  for (int i = 0; i < num_iter; ++i) {
    const int p = i / inner_dim;
    const int q = i % inner_dim;
    float *output_i = output_data + p * channel * inner_dim + q;

    for (int j = 0; j < channel; ++j) {
      if (have_weight) {
        output_i[j * inner_dim] *= weight_data[j];
      }
      if (have_bias) {
        output_i[j * inner_dim] += bias_data[j];
      }
    }
  }

  return success();
}

void top::GroupNormTrainOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  // auto weight_shape = module::getShape(getWeight());
  // auto bias_shape = module::getShape(getBias());
  auto group_num = getNumGroups();
  auto mean_shape = std::vector<int64_t>{input_shape[0], (long)group_num};
  module::setShapeOrVerify(getOutput(), input_shape);
  module::setShapeOrVerify(getMean(), mean_shape);
  module::setShapeOrVerify(getRstd(), mean_shape);
}

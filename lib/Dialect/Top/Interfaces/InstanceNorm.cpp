//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::InstanceNormOp::getFLOPs() {
  const bool have_weight = !module::isNone(getWeight());
  const bool have_bias = !module::isNone(getBias());
  const int64_t num_elem = module::getNumElements(getOutput());
  return num_elem * (10 + have_weight + have_bias);
}

LogicalResult top::InstanceNormOp::init(InferenceParameter &p) {
  return success();
}
void top::InstanceNormOp::deinit(InferenceParameter &p) {}

LogicalResult top::InstanceNormOp::inference(InferenceParameter &p) {
  const float eps_ = getEps().convertToDouble();
  const auto input_shape = module::getShape(getInput());
  const int channel = input_shape[1];

  int outer_dim = 1;
  for (int i = 0; i < 2; i++) {
    outer_dim *= input_shape[i];
  }

  int inner_dim = 1;
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
    const float *input_i = input_data + i * inner_dim;
    float *output_i = output_data + i * inner_dim;
    for (int j = 0; j < inner_dim; ++j) {
      mean_arr[i] += input_i[j];
    }
    mean_arr[i] /= inner_dim;
    for (int j = 0; j < inner_dim; ++j) {
      const float dij = input_i[j] - mean_arr[i];
      rstd_arr[i] += dij * dij;
    }
    rstd_arr[i] /= inner_dim;
    rstd_arr[i] += eps_;
    rstd_arr[i] = std::sqrt(rstd_arr[i]);
    rstd_arr[i] = 1.0f / rstd_arr[i];
    for (int j = 0; j < inner_dim; ++j) {
      output_i[j] = input_i[j] - mean_arr[i];
      output_i[j] *= rstd_arr[i];
    }
  }
  const int num_iter = module::getNumElements(getInput()) / channel;
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

void top::InstanceNormOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  std::vector<int64_t> wb_shape(input_shape.size(), 1);
  wb_shape[1] = input_shape[1];
  RankedTensorType newType;
  if (auto weight_op =
          dyn_cast_or_null<WeightOp>(getWeight().getDefiningOp())) {
    newType =
        RankedTensorType::get(wb_shape, module::getElementType(weight_op));
    getWeight().setType(newType);
  }
  if (auto bias_op = dyn_cast_or_null<WeightOp>(getBias().getDefiningOp())) {
    newType = RankedTensorType::get(wb_shape, module::getElementType(bias_op));
    getBias().setType(newType);
  }
  common_shape_inference(getOperation());
}

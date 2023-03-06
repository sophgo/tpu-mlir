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

int64_t top::LayerNormOp::getFLOPs() {
  const bool have_weight = !getWeight().getType().isa<NoneType>();
  const bool have_bias = !getBias().getType().isa<NoneType>();
  const int64_t num_elem = module::getNumElements(getOutput());
  return num_elem * (10 + have_weight + have_bias);
}

LogicalResult top::LayerNormOp::init(InferenceParameter &p) {
  return success();
}
void top::LayerNormOp::deinit(InferenceParameter &p) {}

LogicalResult top::LayerNormOp::inference(InferenceParameter &p) {
  const int axis_ = getAxis();
  const float eps_ = getEps().convertToDouble();
  const auto input_shape = module::getShape(getInput());

  int outer_dim = 1;
  for (int i = 0; i < axis_; i++) {
    outer_dim *= input_shape[i];
  }

  int inner_dim = 1;
  for (int i = axis_; i < input_shape.size(); i++) {
    inner_dim *= input_shape[i];
  }

  const bool have_weight = !getWeight().getType().isa<NoneType>();
  const bool have_bias = !getBias().getType().isa<NoneType>();
  const bool need_mean = !getMean().getType().isa<NoneType>();
  const bool need_rstd = !getRstd().getType().isa<NoneType>();

  const float *input_data = p.inputs[0];
  const float *weight_data = have_weight ? p.inputs[1] : nullptr;
  const float *bias_data = have_bias ? p.inputs[2] : nullptr;
  float *output_data = p.outputs[0];
  float *mean_data = need_mean ? p.outputs[1] : nullptr;
  float *rstd_data = need_rstd ? p.outputs[2] : nullptr;

  std::vector<float> mean_arr(outer_dim, 0);
  std::vector<float> rstd_arr(outer_dim, 0);

#pragma omp parallel for schedule(static, omp_schedule(outer_dim))
  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < inner_dim; ++j) {
      mean_arr[i] += input_data[i * inner_dim + j];
    }
    mean_arr[i] /= inner_dim;
    if (need_mean) {
      mean_data[i] = mean_arr[i];
    }
    for (int j = 0; j < inner_dim; ++j) {
      const float dij = input_data[i * inner_dim + j] - mean_arr[i];
      rstd_arr[i] += dij * dij;
    }
    rstd_arr[i] /= inner_dim;
    rstd_arr[i] += eps_;
    rstd_arr[i] = std::sqrt(rstd_arr[i]);
    rstd_arr[i] = 1.0f / rstd_arr[i];
    if (need_rstd) {
      rstd_data[i] = rstd_arr[i];
    }
    for (int j = 0; j < inner_dim; ++j) {
      output_data[i * inner_dim + j] =
          input_data[i * inner_dim + j] - mean_arr[i];
      output_data[i * inner_dim + j] *= rstd_arr[i];
      if (have_weight) {
        output_data[i * inner_dim + j] *= weight_data[j];
      }
      if (have_bias) {
        output_data[i * inner_dim + j] += bias_data[j];
      }
    }
  }
  return success();
}

void top::LayerNormOp::shape_inference() {}

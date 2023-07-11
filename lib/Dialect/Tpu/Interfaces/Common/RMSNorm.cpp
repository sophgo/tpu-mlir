//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

static void normlize_f32(const float *input_data, float *output_data,
                         float &rstd_data, const float *gamma_data,
                         const int inner_dim, const float eps) {
  for (int j = 0; j < inner_dim; ++j) {
    const float dij = input_data[j];
    rstd_data += dij * dij;
  }
  rstd_data /= inner_dim;
  rstd_data += eps;
  rstd_data = std::sqrt(rstd_data);

  for (int j = 0; j < inner_dim; ++j) {
    output_data[j] = input_data[j] / rstd_data;
    if (gamma_data) {
      output_data[j] *= gamma_data[j];
    }
  }
}

static void normlize_bf16(const float *input_data, float *output_data,
                          float &rstd_data, const float *gamma_data,
                          const int inner_dim, const float eps) {

  float avg_const = BF16(1.0 / inner_dim);
  for (int j = 0; j < inner_dim; ++j) {
    const float dij = BF16(input_data[j]);
    rstd_data += BF16(BF16(std::pow(dij, 2)) * avg_const);
  }
  rstd_data = BF16(BF16(rstd_data) + BF16(eps));
  rstd_data = BF16(std::sqrt(BF16(rstd_data)));
  rstd_data = BF16(1.0f / rstd_data);

  for (int j = 0; j < inner_dim; ++j) {
    output_data[j] = BF16(input_data[j]);
    output_data[j] = BF16(output_data[j] * rstd_data);
    if (gamma_data) {
      output_data[j] = BF16(output_data[j] * gamma_data[j]);
    }
  }
}

LogicalResult tpu::RMSNormOp::init(InferenceParameter &p) { return success(); }
void tpu::RMSNormOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RMSNormOp::inference(InferenceParameter &p) {
  const float eps = getEps().convertToDouble();
  const auto input_shape = module::getShape(getInput());
  const int dim_size = input_shape.size();
  auto out_type = module::getStorageType(getOutput());
  auto is_bf16 = out_type.isBF16();

  int outer_dim = 1;
  for (int i = 0; i < dim_size - 1; i++) {
    outer_dim *= input_shape[i];
  }
  int inner_dim = input_shape[dim_size - 1];

  const bool have_gamma = !getGamma().getType().isa<NoneType>();

  const float *input_data = p.inputs[0];
  const float *gamma_data = have_gamma ? p.inputs[1] : nullptr;
  float *output_data = p.outputs[0];

  std::vector<float> rms_arr(outer_dim, 0);

#pragma omp parallel for schedule(static, omp_schedule(outer_dim))
  for (int i = 0; i < outer_dim; ++i) {
    float _rstd_data = 0;
    if (is_bf16) {
      normlize_bf16(input_data + i * inner_dim, output_data + i * inner_dim,
                    _rstd_data, gamma_data, inner_dim, eps);
    } else {
      normlize_f32(input_data + i * inner_dim, output_data + i * inner_dim,
                   _rstd_data, gamma_data, inner_dim, eps);
    }
  }

  return success();
}

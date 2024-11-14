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

  const bool has_weight = !getGamma().getType().isa<NoneType>();

  const float *input_data = p.inputs[0];
  const float *gamma_data = has_weight ? p.inputs[1] : nullptr;
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

LogicalResult tpu::RMSNormOp::LocalGenSupport() {
  if (module::isCV18xx() == false) {
    int64_t axis = module::getShape(getInput()).size() - 1;
    // local layer only supports 5 dim at most
    if (axis > 0 && axis <= 4)
      return success();
    else
      return failure();
  }
  return failure();
}

LogicalResult tpu::RMSNormOp::AllowDataSplit(int64_t axis,
                                             group_type_t group_type) {
  int64_t ax = module::getShape(getInput()).size() - 1;
  if (group_type == GROUP_SMALL_C) {
    ax = 2;
  }
  return axis < ax ? success() : failure();
}

bool tpu::RMSNormOp::support_multi_core() { return false; }

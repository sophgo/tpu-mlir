//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/LutFunc.h"

static void normlize_f32(const float *input_data, float *output_data,
                         const float *weight_data, const float *bias_data,
                         const int inner_dim, const float eps_) {
  // The calculation process simulates cmodel
  float mean_data = 0;
  float mean_data_odd = 0;
  float mean_data_even = 0;
  float rstd_data = 0;
  float rstd_data_odd = 0;
  float rstd_data_even = 0;
  float scale = 1.0 / inner_dim;
  for (int j = 0; j < inner_dim; ++j) {
    if (j % 2 == 0) {
      mean_data_even += input_data[j] * scale;
    } else {
      mean_data_odd += input_data[j] * scale;
    }
  }
  mean_data = mean_data_odd + mean_data_even;
  for (int j = 0; j < inner_dim; ++j) {
    const float dij = input_data[j] - mean_data;
    if (j % 2 == 0) {
      rstd_data_even += dij * dij * scale;
    } else {
      rstd_data_odd += dij * dij * scale;
    }
  }
  rstd_data = rstd_data_even + rstd_data_odd;
  rstd_data += eps_;
  rstd_data = std::sqrt(rstd_data);
  rstd_data = 1.0f / rstd_data;
  for (int j = 0; j < inner_dim; ++j) {
    output_data[j] = input_data[j] - mean_data;
    output_data[j] *= rstd_data;
  }
}

static void normlize_bf16(const float *input_data, float *output_data,
                          const float *weight_data, const float *bias_data,
                          float *table, float *mantissa_table,
                          const int inner_dim, const float eps_) {
  float mean = 0;
  float rstd = 0;
  float avg_const = BF16(1.0f / inner_dim);
  for (int j = 0; j < inner_dim; ++j) {
    mean += input_data[j] * avg_const;
  }
  mean = BF16(mean);
  for (int j = 0; j < inner_dim; ++j) {
    const float dij = BF16(input_data[j] - mean);
    rstd += BF16(std::pow(dij, 2)) * avg_const;
  }
  rstd = BF16(BF16(rstd) + BF16(eps_));
  if (module::isCV18xx()) {
    bf16_lut_mantissa(&rstd, &rstd, 1, table, mantissa_table, "mantissa");
  } else {
    rstd = std::sqrt(rstd);
    rstd = 1. / rstd;
  }
  for (int j = 0; j < inner_dim; ++j) {
    output_data[j] = BF16(input_data[j] - mean);
    output_data[j] = BF16(output_data[j] * rstd);
  }
}

LogicalResult tpu::InstanceNormOp::init(InferenceParameter &p) {
  return success();
}
void tpu::InstanceNormOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::InstanceNormOp::inference(InferenceParameter &p) {
  const float eps_ = getEps().convertToDouble();
  const auto input_shape = module::getShape(getInput());
  int channel = input_shape[1];
  auto out_type = module::getStorageType(getOutput());
  auto is_bf16 = out_type.isBF16();

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
  float *table = p.inputs[3];
  float *mtable = p.inputs[4];
  float *output_data = p.outputs[0];

#pragma omp parallel for schedule(static, omp_schedule(outer_dim))
  for (int i = 0; i < outer_dim; ++i) {
    const float *input_i = input_data + i * inner_dim;
    float *output_i = output_data + i * inner_dim;
    if (is_bf16) {
      normlize_bf16(input_i, output_i, weight_data, bias_data, table, mtable,
                    inner_dim, eps_);
    } else {
      normlize_f32(input_i, output_i, weight_data, bias_data, inner_dim, eps_);
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
        if (is_bf16) {
          output_i[j * inner_dim] =
              BF16(output_i[j * inner_dim] * weight_data[j]);
        } else {
          output_i[j * inner_dim] *= weight_data[j];
        }
      }
      if (have_bias) {
        if (is_bf16) {
          output_i[j * inner_dim] =
              BF16(output_i[j * inner_dim] + bias_data[j]);
        } else {
          output_i[j * inner_dim] += bias_data[j];
        }
      }
    }
  }
  return success();
}

LogicalResult tpu::InstanceNormOp::LocalGenSupport() { return success(); }

LogicalResult tpu::InstanceNormOp::AllowDataSplit(int64_t axis,
                                                  group_type_t group_type) {
  return axis < 1 ? success() : failure();
}

bool tpu::InstanceNormOp::support_multi_core() { return false; }

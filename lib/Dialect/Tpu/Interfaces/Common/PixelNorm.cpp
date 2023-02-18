//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"

#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/LutFunc.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

static void normlize_f32(const float *input_data, float *output_data,
                         const float *weight_data, const float *bias_data,
                         const int inner_dim, const int channel, const float eps_) {
  float mean_data = 0;
  float rstd_data = 0;
  for (int j = 0; j < channel; ++j) {
    mean_data += input_data[j * inner_dim];
  }
  mean_data /= channel;
  for (int j = 0; j < channel; ++j) {
    const float dij = input_data[j * inner_dim] - mean_data;
    rstd_data += dij * dij;
  }
  rstd_data /= channel;
  rstd_data += eps_;
  rstd_data = std::sqrt(rstd_data);
  rstd_data = 1.0f / rstd_data;
  for (int j = 0; j < channel; ++j) {
    output_data[j * inner_dim] = input_data[j * inner_dim] - mean_data;
    output_data[j * inner_dim] *= rstd_data;
    if (weight_data) {
      output_data[j * inner_dim] *= weight_data[j];
    }
    if (bias_data) {
      output_data[j * inner_dim] += bias_data[j];
    }
  }
}

static void normlize_bf16(const float *input_data, float *output_data,
                          const float *weight_data, const float *bias_data,
                          float *table, float *mantissa_table,
                          const int inner_dim, const int channel, const float eps_) {
  float mean = 0;
  float rstd = 0;
  float avg_const = BF16(1.0f / channel);
  for (int j = 0; j < channel; ++j) {
    mean += input_data[j * inner_dim] * avg_const;
  }
  mean = BF16(mean);
  for (int j = 0; j < channel; ++j) {
    const float dij = BF16(input_data[j * inner_dim] - mean);
    rstd += BF16(std::pow(dij, 2)) * avg_const;
  }
  rstd = BF16(BF16(rstd) + BF16(eps_));
  if (module::isCV18xx()) {
    bf16_lut_mantissa(&rstd, &rstd, 1, table, mantissa_table,
                      "mantissa");
  } else {
    rstd = std::sqrt(rstd);
    rstd = 1. / rstd;
  }

  for (int j = 0; j < channel; ++j) {
    output_data[j * inner_dim] = BF16(input_data[j * inner_dim] - mean);
    output_data[j * inner_dim] = BF16(output_data[j * inner_dim] * rstd);
    if (weight_data) {
      output_data[j * inner_dim] = BF16(output_data[j * inner_dim] * weight_data[j]);
    }
    if (bias_data) {
      output_data[j * inner_dim] = BF16(output_data[j * inner_dim] + bias_data[j]);
    }
  }
}

LogicalResult tpu::PixelNormOp::init(InferenceParameter &p) {
  return success();
}
void tpu::PixelNormOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::PixelNormOp::inference(InferenceParameter &p) {
  const float eps_ = getEps().convertToDouble();
  const auto input_shape = module::getShape(getInput());
  auto out_type = module::getStorageType(getOutput());
  auto is_bf16 = out_type.isBF16();

  int outer_dim = input_shape[0];
  int channel = input_shape[1];
  int inner_dim = 1;
  for (int i = 2; i < input_shape.size(); i++) {
    inner_dim *= input_shape[i];
  }

  const bool have_weight = !getWeight().getType().isa<mlir::NoneType>();
  const bool have_bias = !getBias().getType().isa<mlir::NoneType>();

  const float *input_data = p.inputs[0];
  const float *weight_data = have_weight ? p.inputs[1] : nullptr;
  const float *bias_data = have_bias ? p.inputs[2] : nullptr;
  float *table = p.inputs[3];
  float *mtable = p.inputs[4];
  float *output_data = p.outputs[0];

  const int num_iter = outer_dim * inner_dim;
  //#pragma omp parallel for schedule(static, omp_schedule(num_iter))
  for (int i = 0; i < num_iter; ++i) {
    const int p = i / inner_dim;
    const int q = i % inner_dim;
    const float* input_i = input_data + p * channel * inner_dim + q;
    float* output_i = output_data + p * channel * inner_dim + q;
    if (is_bf16) {
      normlize_bf16(input_i, output_i, weight_data, bias_data, table,
                    mtable, inner_dim, channel, eps_);
    } else {
      normlize_f32(input_i, output_i, weight_data, bias_data, inner_dim,
                   channel, eps_);
    }
  }
  return success();
}

// TODO: activate it later
LogicalResult tpu::PixelNormOp::LocalGenSupport() {
  return success();
}

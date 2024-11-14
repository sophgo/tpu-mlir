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
                         const int inner_dim, const int channel,
                         const float eps_) {
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
                          const int inner_dim, const int channel,
                          const float eps_) {
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
    bf16_lut_mantissa(&rstd, &rstd, 1, table, mantissa_table, "mantissa");
  } else {
    rstd = std::sqrt(rstd);
    rstd = 1. / rstd;
  }

  for (int j = 0; j < channel; ++j) {
    output_data[j * inner_dim] = BF16(input_data[j * inner_dim] - mean);
    output_data[j * inner_dim] = BF16(output_data[j * inner_dim] * rstd);
    if (weight_data) {
      output_data[j * inner_dim] =
          BF16(output_data[j * inner_dim] * weight_data[j]);
    }
    if (bias_data) {
      output_data[j * inner_dim] =
          BF16(output_data[j * inner_dim] + bias_data[j]);
    }
  }
}

template <typename T>
T sadd(T a, T b) {
  static_assert(std::is_integral<T>::value,
                "sadd is not defined for non-integral types");
  const T max_val = std::numeric_limits<T>::max();
  const T min_val = std::numeric_limits<T>::min();
  if (a > 0) {
    if (b > max_val - a) {
      return max_val;
    }
  } else {
    if (b < min_val - a) {
      return min_val;
    }
  }
  return a + b;
}

static void normlize_i8(const float *input, float *output, const float *weight,
                        const float *bias, int inner_dim, int channel,
                        float eps, float scale, bool is_signed) {
  const float avg_const = F16(1.0f / channel);
  const float eps_f16 = F16(eps);
  const float scale_f16 = F16(scale);
  const float scale_sq_f16 = F16(scale_f16 * scale_f16);
  int32_t sum_x = 0, sum_x2 = 0;
  for (int j = 0; j < channel; ++j) {
    if (is_signed) {
      int8_t x = input[j * inner_dim];
      sum_x = sadd<int32_t>(sum_x, x);
      sum_x2 = sadd<int32_t>(sum_x2, x * x);
    } else {
      uint8_t x = input[j * inner_dim];
      sum_x = sadd<int32_t>(sum_x, x);
      sum_x2 = sadd<int32_t>(sum_x2, x * x);
    }
  }
  float mean = F16(F16((float)sum_x) * avg_const);
  float rstd = F16(F16((float)sum_x2) * avg_const);
  float mean_sq = F16(mean * mean);
  rstd = F16(rstd - mean_sq);
  rstd = std::max(rstd, 0.0f);
  rstd = F16(rstd * scale_sq_f16);
  rstd = F16(rstd + eps_f16);
  rstd = F16(1.0f / sqrtf(rstd));
  rstd = F16(rstd * scale_f16);
  for (int j = 0; j < channel; ++j) {
    output[j * inner_dim] = F16(input[j * inner_dim] - mean);
    output[j * inner_dim] = F16(output[j * inner_dim] * rstd);
    if (weight)
      output[j * inner_dim] = F16(output[j * inner_dim] * weight[j]);
    if (bias)
      output[j * inner_dim] = F16(output[j * inner_dim] + bias[j]);
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
#pragma omp parallel for schedule(static, omp_schedule(num_iter))
  for (int i = 0; i < num_iter; ++i) {
    const int p = i / inner_dim;
    const int q = i % inner_dim;
    const float *input_i = input_data + p * channel * inner_dim + q;
    float *output_i = output_data + p * channel * inner_dim + q;
    if (!module::isUniformQuantized(getInput())) {
      if (is_bf16) {
        normlize_bf16(input_i, output_i, weight_data, bias_data, table, mtable,
                      inner_dim, channel, eps_);
      } else {
        normlize_f32(input_i, output_i, weight_data, bias_data, inner_dim,
                     channel, eps_);
      }
    } else {
      const auto qtype = module::getUniformQuantizedType(getInput());
      normlize_i8(input_i, output_i, weight_data, bias_data, inner_dim, channel,
                  eps_, qtype.getScale(), qtype.isSigned());
    }
  }
  return success();
}

LogicalResult tpu::PixelNormOp::LocalGenSupport() { return success(); }

mlir::Type tpu::PixelNormOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  auto op = getOperation();
  if (opd_idx == 0) {
    auto opd = op->getOperand(0);
    auto in_op = opd.getDefiningOp();
    if (in_op != nullptr && isa<top::WeightOp, top::NoneOp>(in_op)) {
      return do_nothing(mode);
    }
    if (module::isUniformQuantized(opd)) {
      mode = TypeCastMode::DO_NOTHING;
      return Builder(op).getIntegerType(8);
    }
  }
  return type_verify_case_same(op, opd_idx, mode);
}

bool tpu::PixelNormOp::support_multi_core() { return false; }

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
                         float &mean_data, float &rstd_data,
                         const float *weight_data, const float *bias_data,
                         const int inner_dim, const float eps_) {
  for (int j = 0; j < inner_dim; ++j) {
    mean_data += input_data[j];
  }
  mean_data /= inner_dim;

  for (int j = 0; j < inner_dim; ++j) {
    const float dij = input_data[j] - mean_data;
    rstd_data += dij * dij;
  }
  rstd_data /= inner_dim;
  rstd_data += eps_;
  rstd_data = std::sqrt(rstd_data);
  rstd_data = 1.0f / rstd_data;

  for (int j = 0; j < inner_dim; ++j) {
    output_data[j] = input_data[j] - mean_data;
    output_data[j] *= rstd_data;
    if (weight_data) {
      output_data[j] *= weight_data[j];
    }
    if (bias_data) {
      output_data[j] += bias_data[j];
    }
  }
}

static void normlize_bf16(const float *input_data, float *output_data,
                          float &mean_data, float &rstd_data,
                          const float *weight_data, const float *bias_data,
                          float *table, float *mantissa_table,
                          const int inner_dim, const float eps_) {

  float avg_const = BF16(1.0 / inner_dim);
  for (int j = 0; j < inner_dim; ++j) {
    mean_data += input_data[j] * avg_const;
  }
  mean_data = BF16(mean_data);
  for (int j = 0; j < inner_dim; ++j) {
    const float dij = BF16(input_data[j] - mean_data);
    rstd_data += BF16(BF16(std::pow(dij, 2)) * avg_const);
  }
  rstd_data = BF16(BF16(rstd_data) + BF16(eps_));
  if (module::isCV18xx()) {
    bf16_lut_mantissa(&rstd_data, &rstd_data, 1, table, mantissa_table,
                      "mantissa");
  } else {
    rstd_data = BF16(std::sqrt(BF16(rstd_data)));
    rstd_data = BF16(1.0f / rstd_data);
  }

  for (int j = 0; j < inner_dim; ++j) {
    output_data[j] = BF16(input_data[j] - mean_data);
    output_data[j] = BF16(output_data[j] * rstd_data);
    if (weight_data) {
      output_data[j] = BF16(output_data[j] * weight_data[j]);
    }
    if (bias_data) {
      output_data[j] = BF16(output_data[j] + bias_data[j]);
    }
  }
}

LogicalResult tpu::LayerNormOp::init(InferenceParameter &p) {
  return success();
}
void tpu::LayerNormOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::LayerNormOp::inference(InferenceParameter &p) {
  const int axis_ = getAxis();
  const float eps_ = getEps().convertToDouble();
  const auto input_shape = module::getShape(getInput());
  auto out_type = module::getStorageType(getOutput());
  auto is_bf16 = out_type.isBF16();

  int outer_dim = 1;
  for (int i = 0; i < axis_; i++) {
    outer_dim *= input_shape[i];
  }

  int inner_dim = 1;
  for (int i = axis_; i < input_shape.size(); i++) {
    inner_dim *= input_shape[i];
  }

  const bool have_weight = !getWeight().getType().isa<mlir::NoneType>();
  const bool have_bias = !getBias().getType().isa<mlir::NoneType>();
  const bool need_mean = !getMean().getType().isa<mlir::NoneType>();
  const bool need_rstd = !getRstd().getType().isa<mlir::NoneType>();

  const float *input_data = p.inputs[0];
  const float *weight_data = have_weight ? p.inputs[1] : nullptr;
  const float *bias_data = have_bias ? p.inputs[2] : nullptr;
  float *table = p.inputs[3];
  float *mtable = p.inputs[4];
  float *output_data = p.outputs[0];
  float *mean_data = need_mean ? p.outputs[1] : nullptr;
  float *rstd_data = need_rstd ? p.outputs[2] : nullptr;

  std::vector<float> mean_arr(outer_dim, 0);
  std::vector<float> rstd_arr(outer_dim, 0);

  for (int i = 0; i < outer_dim; ++i) {
    float _mean_data = 0;
    float _rstd_data = 0;
    if (is_bf16) {
      normlize_bf16(input_data + i * inner_dim, output_data + i * inner_dim,
                    _mean_data, _rstd_data, weight_data, bias_data, table,
                    mtable, inner_dim, eps_);
    } else {
      normlize_f32(input_data + i * inner_dim, output_data + i * inner_dim,
                   _mean_data, _rstd_data, weight_data, bias_data, inner_dim,
                   eps_);
    }
    if (need_mean) {
      mean_data[i] = _mean_data;
    }
    if (need_rstd) {
      rstd_data[i] = _rstd_data;
    }
  }
  return success();
}

LogicalResult tpu::LayerNormOp::LocalGenSupport() {
  if (module::isCV18xx() == false) {
    if (getAxis() != 0)
      return success();
    else
      return failure();
  }
  return failure();
}

LogicalResult tpu::LayerNormOp::AllowDataSplit(int64_t axis,
                                               group_type_t group_type) {
  int64_t ax = getAxis();
  if (group_type == GROUP_SMALL_C) {
    assert(ax == 4);
    ax = 2;
  }
  return axis < ax ? success() : failure();
}

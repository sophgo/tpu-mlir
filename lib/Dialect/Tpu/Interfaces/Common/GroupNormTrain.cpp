//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/LutFunc.h"

static void normlize_f32(const float *input_data, float *output_data,
                         const int inner_dim, const float eps_, float *mean_arr,
                         float *rstd_arr) {
  float mean_data = 0;
  float rstd_data = 0;
  for (int j = 0; j < inner_dim; ++j) {
    mean_data += (input_data[j] / inner_dim);
  }
  for (int j = 0; j < inner_dim; ++j) {
    const float dij = input_data[j] - mean_data;
    rstd_data += (dij * dij / inner_dim);
  }
  rstd_data += eps_;
  rstd_data = std::sqrt(rstd_data);
  rstd_data = 1.0f / rstd_data;
  for (int j = 0; j < inner_dim; ++j) {
    output_data[j] = input_data[j] - mean_data;
    output_data[j] *= rstd_data;
  }
  *mean_arr = mean_data;
  *rstd_arr = rstd_data;
}

static void normlize_bf16(const float *input_data, float *output_data,
                          float *table, float *mantissa_table,
                          const int inner_dim, const float eps_,
                          float *mean_arr, float *rstd_arr) {
  float mean = BF16(0);
  float rstd = BF16(0);
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
  rstd = std::sqrt(rstd);
  rstd = 1. / rstd;
  for (int j = 0; j < inner_dim; ++j) {
    output_data[j] = BF16(input_data[j] - mean);
    output_data[j] = BF16(output_data[j] * rstd);
  }
  *mean_arr = BF16(mean);
  *rstd_arr = BF16(rstd);
}

LogicalResult tpu::GroupNormTrainOp::init(InferenceParameter &p) {

  return success();
}

void tpu::GroupNormTrainOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::GroupNormTrainOp::inference(InferenceParameter &p) {
  const float eps_ = getEps().convertToDouble();
  const auto input_shape = module::getShape(getInput());
  const int channel = input_shape[1];
  const int num_groups = getNumGroups();
  ASSERT_THIS(channel % num_groups == 0);
  const int channel_per_group = channel / num_groups;
  auto out_type = module::getStorageType(getOutput());
  auto is_bf16 = out_type.isBF16();

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
  float *table = p.inputs[3];
  float *mtable = p.inputs[4];
  float *output_data = p.outputs[0];
  float *mean_arr = p.outputs[1];
  float *rstd_arr = p.outputs[2];

#pragma omp parallel for schedule(static, omp_schedule(outer_dim))
  for (int i = 0; i < outer_dim; ++i) {
    const float *input_i = input_data + i * inner_dim;
    float *output_i = output_data + i * inner_dim;
    float *mean_i = mean_arr + i;
    float *rstd_i = rstd_arr + i;
    if (is_bf16) {
      normlize_bf16(input_i, output_i, table, mtable, inner_dim, eps_, mean_i,
                    rstd_i);
    } else {
      normlize_f32(input_i, output_i, inner_dim, eps_, mean_i, rstd_i);
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
      float *output_ij = output_i + j * inner_dim;
      if (have_weight) {
        *output_ij *= weight_data[j];
      }
      if (have_bias) {
        *output_ij += bias_data[j];
      }
    }
  }

  return success();
}

uint32_t tpu::GroupNormTrainOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}

int64_t tpu::GroupNormTrainOp::dyn_codegen_global_bm1684x(void *ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}

int64_t tpu::GroupNormTrainOp::get_fw_type_bm1684() { return -1; }

int64_t tpu::GroupNormTrainOp::get_fw_type_bm1684x() { return -1; }

bool tpu::GroupNormTrainOp::support_multi_core() { return false; }

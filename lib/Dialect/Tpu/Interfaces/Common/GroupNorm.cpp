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
  float mean_data = 0;
  float rstd_data = 0;
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

LogicalResult tpu::GroupNormOp::init(InferenceParameter &p) {
  return success();
}
void tpu::GroupNormOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::GroupNormOp::inference(InferenceParameter &p) {
  const float eps_ = getEps().convertToDouble();
  const auto input_shape = module::getShape(getInput());
  const int channel = input_shape[1];
  const int num_groups = getNumGroups();
  assert(channel % num_groups == 0);
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

LogicalResult tpu::GroupNormOp::LocalGenSupport() {
  if (module::isBM1684Family()) {
    auto input_shape = module::getShape(getInput());
    int num_dims = input_shape.size();
    int input_c = num_dims > 1 ? input_shape[1] : 1;
    int input_w = num_dims > 3 ? input_shape[3] : 1;
    bool support_c = input_c < (((int)1) << 12) && input_c >= 0;
    bool support_w = input_w < (((int)1) << 16) && input_w >= 0;
    if (!support_c || !support_w) {
      return failure();
    }
  }
  if (module::isBM1690Family()) {
    return failure();
  }
  return success();
}

LogicalResult tpu::GroupNormOp::AllowDataSplit(int64_t axis,
                                               group_type_t group_type) {
  return axis < 1 ? success() : failure();
}

ArrayAttr tpu::GroupNormOp::getIndexingMaps() {
  MLIRContext *context = getContext();

  AffineExpr d0, d1;
  bindDims(context, d0, d1);
  auto c0 = mlir::getAffineConstantExpr(0, context);
  auto inputMap = AffineMap::getMultiDimIdentityMap(2, context);
  auto weightMap = AffineMap::get(2, 0, {c0, d1}, context);
  auto outputMap = AffineMap::getMultiDimIdentityMap(2, context);
  auto empty = AffineMap::get(2, 0, context);

  SmallVector<AffineMap> indexingMaps{inputMap};

  for (int i = 1, n = getNumOperands(); i < n; ++i) {
    if (isa_and_nonnull<top::NoneOp>(getOperand(i).getDefiningOp()))
      indexingMaps.push_back(empty);
    else
      indexingMaps.push_back(weightMap);
  }
  indexingMaps.push_back(outputMap);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
}

bool tpu::GroupNormOp::support_multi_core() { return false; }

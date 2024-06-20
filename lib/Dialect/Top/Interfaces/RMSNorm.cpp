//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::RMSNormOp::getFLOPs() {
  return 3 * module::getNumElements(getOutput());
}

LogicalResult top::RMSNormOp::init(InferenceParameter &p) { return success(); }
void top::RMSNormOp::deinit(InferenceParameter &p) {}

LogicalResult top::RMSNormOp::inference(InferenceParameter &p) {
  const float eps = getEps().convertToDouble();
  const auto input_shape = module::getShape(getInput());
  const int dim_size = input_shape.size();

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
    for (int j = 0; j < inner_dim; ++j) {
      const float dij = input_data[i * inner_dim + j];
      rms_arr[i] += dij * dij;
    }
    rms_arr[i] /= inner_dim;
    rms_arr[i] += eps;
    rms_arr[i] = std::sqrt(rms_arr[i]);

    for (int j = 0; j < inner_dim; ++j) {
      output_data[i * inner_dim + j] =
          input_data[i * inner_dim + j] / rms_arr[i];
      if (have_gamma) {
        output_data[i * inner_dim + j] *= gamma_data[j];
      }
    }
  }

  return success();
}

void top::RMSNormOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  auto dims = in_shape.size();
  if (!module::isNone(getGamma())) {
    auto normalized_shape = module::getShape(getGamma());
    ASSERT_THIS(normalized_shape.size() == 1 &&
                normalized_shape[0] == in_shape[dims - 1]);
  }
  module::setShapeOrVerify(getOutput(), in_shape);

  if (module::isWeight(getGamma())) {
    broadcast_tensor_reshape(getOutput(), getGamma());
  }
}

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::LeakyReluOp::init(InferenceParameter &p) {
  return success();
}
void tpu::LeakyReluOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::LeakyReluOp::inference(InferenceParameter &p) {
  int64_t num_elements = module::getNumElements(getInput());
  memset(p.outputs[0], 0, sizeof(float) * num_elements);
  auto out_type = module::getStorageType(getOutput());
  auto asym = module::isAsymmetric();
  bool is_cv18xx = module::isCV18xx();
  auto round_mode = round_mode_convert(getRoundMode());

  if (out_type.isa<FloatType>()) {
    float *src = p.inputs[0];
    float *dst = p.outputs[0];
    float alpha = static_cast<float>(getAlpha().value().convertToDouble());
#pragma omp parallel for schedule(static, omp_schedule(num_elements))
    for (int64_t i = 0; i < num_elements; ++i) {
      dst[i] = src[i] > 0 ? src[i] : (alpha * src[i]);
    }
    if (out_type.isF16()) {
      F16(dst, dst, num_elements);
    } else if (out_type.isBF16()) {
      BF16(dst, dst, num_elements);
    }
  } else if (asym == false) {
    int64_t scale_neg, shift_neg;
    int64_t scalei = getMultiplier().value();
    int64_t shifti = getRshift().value();
    bool do_pos_scale = false;
    if (is_cv18xx) {
      scale_neg = getMultiplierNeg().value();
      shift_neg = getRshiftNeg().value();
      do_pos_scale = scalei != 0 ? true : false;
    }

#pragma omp parallel for schedule(static, omp_schedule(num_elements))
    for (int64_t i = 0; i < num_elements; ++i) {
      int64_t dst = 0;
      int64_t src = static_cast<int64_t>(p.inputs[0][i]);
      if (is_cv18xx) {
        if (src >= 0) {
          dst = do_pos_scale ? applyMultiplierAndRShift(src, scalei, shifti)
                             : src;
        } else {
          dst = applyMultiplierAndRShift(src, scale_neg, shift_neg);
        }
      } else {
        dst = src >= 0 ? src : applyMultiplierAndRShift(src, scalei, shifti);
      }
      p.outputs[0][i] = saturate(dst, out_type);
    }
  } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elements))
    for (int64_t i = 0; i < num_elements; ++i) {
      int64_t src = static_cast<int64_t>(p.inputs[0][i]);
      int64_t dst = 0;

      auto i_qtype = module::getUniformQuantizedType(getInput());
      auto o_qtype = module::getUniformQuantizedType(getOutput());
      double scale = i_qtype.getScale() / o_qtype.getScale();
      dst = src >= i_qtype.getZeroPoint()
                ? src
                : ((src - i_qtype.getZeroPoint()) * (float)(1.0 / scale) +
                   o_qtype.getZeroPoint());
      p.outputs[0][i] = saturate(dst, out_type, round_mode);
    }
  }
  return success();
}

void tpu::LeakyReluOp::assign_fw_param(void *param) {
  fw_prelu_layer_param_t prelu_param = {0};
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  prelu_param.ic = (uint32_t)c;
  prelu_param.channel_shared = 1;
  prelu_param.relu_upper_limit = -1; // no use
  if (module::isUniformQuantized(getInput())) {
    prelu_param.rshift_bit = getRshift().value();
    prelu_param.in_sign = module::isSign(getInput());
    prelu_param.out_sign = module::isSign(getOutput());
    prelu_param.shared_slope = static_cast<float>(getMultiplier().value());
  } else {
    prelu_param.shared_slope =
        static_cast<float>(getAlpha().value().convertToDouble());
  }
  memcpy(param, &prelu_param, sizeof(fw_prelu_layer_param_t));
}

ArrayAttr tpu::LeakyReluOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::LeakyReluOp::support_multi_core() { return false; }

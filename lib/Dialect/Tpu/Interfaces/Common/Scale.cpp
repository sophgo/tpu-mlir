//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::ScaleOp::init(InferenceParameter &p) { return success(); }
void tpu::ScaleOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ScaleOp::inference(InferenceParameter &p) {
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  const float *src = p.inputs[0];
  const float *scale = p.inputs[1];
  const float *bias = p.inputs[2];
  float *dst = p.outputs[0];

  auto out_type = module::getStorageType(getOutput());
  auto asym = module::isAsymmetric();
  if (out_type.isa<FloatType>()) {
#pragma omp parallel for schedule(static, omp_schedule(c))
    for (int64_t i = 0; i < c; ++i) {
      float scale_val = scale[i];
      float bias_val = bias[i];
      for (int64_t j = 0; j < n; ++j) {
        for (int64_t k = 0; k < h * w; ++k) {
          int64_t idx = j * c * h * w + i * h * w + k;
          dst[idx] = src[idx] * scale_val + bias_val;
        }
      }
    }
  } else if (asym == false) {
    const float *lshift = p.inputs[3];
    auto res_sign = module::isSign(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(c))
    for (int64_t i = 0; i < c; ++i) {
      int32_t scale_val = scale[i];
      int32_t bias_val = bias[i];
      int32_t rshift_val = -lshift[i];
      for (int64_t j = 0; j < n; ++j) {
        for (int64_t k = 0; k < h * w; ++k) {
          int64_t idx = j * c * h * w + i * h * w + k;
          int64_t res = (int64_t)src[idx] * scale_val + bias_val;
          if (module::isBM1684Family()) {
            // sature to INT16/UINT16
            if (res_sign)
              res = to_int16(res);
            else
              res = to_uint16(res);
          }
          res = RightShiftRound(res, rshift_val, ROUNDING_DOWN);
          if (getDoRelu() && res < 0) {
            res = 0;
          }
          dst[idx] = saturate(res, out_type);
        }
      }
    }
  } else {
    const float *lshift = p.inputs[3];
    auto o_qtype = module::getUniformQuantizedType(getOutput());
    int64_t out_zp = o_qtype.getZeroPoint();
#pragma omp parallel for schedule(static, omp_schedule(c))
    for (int64_t i = 0; i < c; ++i) {
      int32_t scale_val = scale[i];
      int32_t bias_val = bias[i];
      int32_t rshift_val = -lshift[i];
      for (int64_t j = 0; j < n; ++j) {
        for (int64_t k = 0; k < h * w; ++k) {
          int64_t idx = j * c * h * w + i * h * w + k;
          int64_t res = (int64_t)src[idx] * scale_val + bias_val;
          res = RightShiftRound(res, rshift_val, ROUNDING_DOWN) + out_zp;
          if (getDoRelu() && res < 0) {
            res = 0;
          }
          dst[idx] = saturate(res, out_type);
        }
      }
    }
  }

  auto num_elem = module::getNumElements(getOutput());
  if (getDoRelu()) {
    auto relu_limit = getReluLimit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0], num_elem, relu_limit);
  }

  return success();
}

ArrayAttr tpu::ScaleOp::getIndexingMaps() {
  MLIRContext *ctx = getContext();
  auto out_shape = module::getShape(getOutput());
  auto num_dims = out_shape.size();
  SmallVector<AffineMap, 8> indexingMaps;
  AffineMap outMap = AffineMap::getMultiDimIdentityMap(num_dims, ctx);
  auto empty_map = AffineMap::get(num_dims, 0, ctx);
  for (auto opd : getOperands()) {
    if (module::isNone(opd)) {
      indexingMaps.emplace_back(empty_map);
      continue;
    }
    auto shape = module::getShape(opd);
    auto map = getBinaryMap(outMap, shape);
    indexingMaps.emplace_back(map);
  }
  indexingMaps.emplace_back(outMap);
  return Builder(ctx).getAffineMapArrayAttr(indexingMaps);
}

bool tpu::ScaleOp::support_multi_core() { return false; }

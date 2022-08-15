//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::ScaleOp::init(InferenceParameter &p) { return success(); }
void tpu::ScaleOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ScaleOp::inference(InferenceParameter &p) {
  auto module = Module::getModuleOp(getOperation());
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  const float *src = p.inputs[0];
  const float *scale = p.inputs[1];
  const float *bias = p.inputs[2];
  float *dst = p.outputs[0];

  auto out_type = Module::getStorageType(output());
  auto asym = Module::getAsymmetric(module);
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
#pragma omp parallel for schedule(static, omp_schedule(c))
    for (int64_t i = 0; i < c; ++i) {
      int32_t scale_val = scale[i];
      int32_t bias_val = bias[i];
      int32_t rshift_val = -lshift[i];
      for (int64_t j = 0; j < n; ++j) {
        for (int64_t k = 0; k < h * w; ++k) {
          int64_t idx = j * c * h * w + i * h * w + k;
          int64_t res = (int64_t)src[idx] * scale_val + bias_val;
          res = RightShiftRound(res, rshift_val, ROUNDING_HALF_UP);
          if (do_relu() && res < 0) {
            res = 0;
          }
          dst[idx] = out_type.isUnsignedInteger(8) ? Quant::to_uint8(res)
                                                   : Quant::to_int8(res);
        }
      }
    }
  } else {
    const float *lshift = p.inputs[3];
    auto o_qtype = Quant::getUniformQuantizedType(output());
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
          res = RightShiftRound(res, rshift_val, ROUNDING_HALF_UP) + out_zp;
          if (do_relu() && res < 0) {
            res = 0;
          }
          dst[idx] = out_type.isUnsignedInteger(8) ? Quant::to_uint8(res)
                                                   : Quant::to_int8(res);
        }
      }
    }
  }

  auto num_elem = Module::getNumElements(output());
  if (do_relu()) {
    function_relu(p.outputs[0], p.outputs[0], num_elem);
  }

  return success();
}


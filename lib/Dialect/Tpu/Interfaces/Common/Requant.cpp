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
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::RequantOp::init(InferenceParameter &p) { return success(); }
void tpu::RequantOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RequantOp::inference(InferenceParameter &p) {
  auto i_sType = Module::getStorageType(output());
  auto o_sType = Module::getStorageType(output());
  auto i_qtype = Quant::getUniformQuantizedType(input());
  auto o_qtype = Quant::getUniformQuantizedType(output());
  if (i_sType.isInteger(8)) {
    int64_t n, c, h, w;
    Module::getNCHW(output(), n, c, h, w);
    std::shared_ptr<std::vector<int32_t>> quant_v;
    std::vector<int64_t> quant_shape = {1, c, 1, 3};
    bool per_axis = false;
    if (quant().getType().isa<RankedTensorType>()) {
      auto quantOp = quant().getDefiningOp<top::WeightOp>();
      quant_v = quantOp.read<int32_t>();
      per_axis = true;
    }
    int64_t rshift_val = rshift().getValue();
    int64_t mul_val = multiplier().getValue();
    int64_t zp_x = i_qtype.getZeroPoint();
#pragma omp parallel for schedule(static, omp_schedule(c))
    for (int ic = 0; ic < c; ic++) {
      int64_t multi = per_axis ? quant_v->at(ic * 3 + 1) : mul_val;
      int64_t r_shift = per_axis ? quant_v->at(ic * 3) : rshift_val;
      int64_t zp_y =
          per_axis ? quant_v->at(ic * 3 + 2) : o_qtype.getZeroPoint();
      for (int in = 0; in < n; in++) {
        for (int hw = 0; hw < h * w; hw++) {
          int64_t idx = (in * c + ic) * h * w + hw;
          int32_t tmp = (int32_t)p.inputs[0][idx] - zp_x;
          auto v = applyMultiplierAndRShift(tmp, (int32_t)multi, (int32_t)r_shift) + zp_y;
          p.outputs[0][idx] = o_sType.isUnsignedInteger(8) ? Quant::to_uint8(v)
                                                           : Quant::to_int8(v);
        }
      }
    }
    return success();
  }

  auto shape = Module::getShape(output());
  auto mode = quant_mode().getValue();
  int64_t inner = 1;
  for (int i = 2; i < shape.size(); ++i) {
    inner *= shape[i];
  }
  auto multi = multiplier().getValue();
  auto rshift_val = rshift().getValue();
  auto zero_point = o_qtype.getZeroPoint();
  if (mode == 0) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      if (quant().getType().isa<NoneType>()) {
        rshift_val = -p.inputs[1][c * 3 + 1];
        multi = p.inputs[1][c * 3];
        zero_point = p.inputs[1][c * 3 - 2];
      }
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          p.outputs[0][offset] =
              zero_point + MultiplyByQuantizedMultiplier(
                               (int32_t)p.inputs[0][offset], (int32_t)multi,
                               (int32_t)rshift_val);
        }
      }
    }
  } else if (mode == 1) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      if (quant().getType().isa<NoneType>()) {
        rshift_val = -p.inputs[1][c * 3 + 1];
        multi = p.inputs[1][c * 3];
        zero_point = p.inputs[1][c * 3 - 2];
      }
      assert(rshift_val > 0);
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          p.outputs[0][offset] =
              zero_point + MultiplyByQuantizedMultiplier(
                               (int32_t)p.inputs[0][offset], (int32_t)multi,
                               (int32_t)rshift_val);
        }
      }
    }
  } else if (mode == 2) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      if (quant().getType().isa<NoneType>()) {
        rshift_val = -p.inputs[1][c * 3 + 1];
        multi = p.inputs[1][c * 3];
        zero_point = p.inputs[1][c * 3 - 2];
      }
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          p.outputs[0][offset] =
              zero_point +
              applyMultiplierAndRShift(p.inputs[0][offset], multi, rshift_val);
        }
      }
    }
  }
  return success();
}

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
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::RequantOp::init(InferenceParameter &p) { return success(); }
void tpu::RequantOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RequantOp::inference(InferenceParameter &p) {
  auto shape = Module::getShape(output());
  auto mode = quant_mode().getValue();
  int64_t inner = 1;
  for (int i = 2; i < shape.size(); ++i) {
    inner *= shape[i];
  }
  auto qtype = Quant::getUniformQuantizedType(output());
  auto multi = multiplier().getValue();
  auto shift = rshift().getValue();
  auto zero_point = qtype.getZeroPoint();
  if (mode == 0) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      if (quant().getType().isa<NoneType>()) {
        shift = p.inputs[1][c * 3 + 1];
        multi = p.inputs[1][c * 3];
        zero_point = p.inputs[1][c * 3 - 2];
      }
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          p.outputs[0][offset] = zero_point +
              MultiplyByQuantizedMultiplier((int32_t)p.inputs[0][offset],
                                            (int32_t)multi, (int32_t)shift);
        }
      }
    }
  } else if (mode == 1) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      if (quant().getType().isa<NoneType>()) {
        shift = p.inputs[1][c * 3 + 1];
        multi = p.inputs[1][c * 3];
        zero_point = p.inputs[1][c * 3 - 2];
      }
      assert(shift < 0);
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          p.outputs[0][offset] = zero_point +
              MultiplyByQuantizedMultiplier((int32_t)p.inputs[0][offset],
                                            (int32_t)multi, (int32_t)shift);
        }
      }
    }
  } else if (mode == 2) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      if (quant().getType().isa<NoneType>()) {
        shift = p.inputs[1][c * 3 + 1];
        multi = p.inputs[1][c * 3];
        zero_point = p.inputs[1][c * 3 - 2];
      }
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          p.outputs[0][offset] = zero_point +
              applyMultiplierAndRShift(p.inputs[0][offset], multi, shift);
        }
      }
    }
  }
  return success();
}

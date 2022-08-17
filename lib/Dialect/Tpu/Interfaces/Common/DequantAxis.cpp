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

LogicalResult tpu::DequantAxisOp::init(InferenceParameter &p) {
  return success();
}
void tpu::DequantAxisOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::DequantAxisOp::inference(InferenceParameter &p) {
  auto i_sType = Module::getStorageType(input());
  auto o_sType = Module::getStorageType(output());
  auto o_qtype = Quant::getUniformQuantizedType(output());

  auto shape = Module::getShape(output());
  auto mode = quant_mode();
  int64_t inner = 1;
  for (int i = 2; i < shape.size(); ++i) {
    inner *= shape[i];
  }

  int32_t *quant_p = reinterpret_cast<int32_t*>(p.inputs[1]);
  int32_t *output_p = reinterpret_cast<int32_t*>(p.outputs[0]);
  if (mode == 0) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      int64_t multi = quant_p[c * 3];
      int64_t shift_val = quant_p[c * 3 + 1];
      int64_t zero_point = quant_p[c * 3 + 2];
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          int32_t tmp = (int32_t)p.inputs[0][offset] - zero_point;
          output_p[offset] = applyMultiplierAndRShift(tmp, multi, -shift_val);
        }
      }
    }
  } else if (mode == 1) {
    int64_t lshift_val = lshift().getValue();
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      int64_t multi = quant_p[c * 3];
      int64_t shift_val = quant_p[c * 3 + 1];
      int64_t zero_point = quant_p[c * 3 + 2];
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          int64_t tmp = ((int32_t)p.inputs[0][offset] - zero_point) << lshift_val;
          output_p[offset] = MultiplyByQuantizedMultiplier(tmp, 1, -shift_val);
        }
      }
    }
  } else {
    llvm_unreachable("no such dequant mode");
  }
  return success();
}

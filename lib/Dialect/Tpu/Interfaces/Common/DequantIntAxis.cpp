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
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::DequantIntAxisOp::init(InferenceParameter &p) {
  return success();
}
void tpu::DequantIntAxisOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::DequantIntAxisOp::inference(InferenceParameter &p) {
  auto i_sType = Module::getStorageType(input());
  auto o_sType = Module::getStorageType(output());
  auto o_qtype = Quant::getUniformQuantizedType(output());

  auto shape = Module::getShape(output());
  auto mode = quant_mode();
  int64_t inner = 1;
  for (int i = 2; i < shape.size(); ++i) {
    inner *= shape[i];
  }

  if (mode == DequantMode::Normal) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      int64_t multi = p.inputs[1][c * 3];
      int64_t shift_val = p.inputs[1][c * 3 + 1];
      int64_t zero_point = p.inputs[1][c * 3 + 2];
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          int32_t tmp = (int32_t)p.inputs[0][offset] - zero_point;
          p.outputs[0][offset] = applyMultiplierAndRShift(tmp, multi, -shift_val);
        }
      }
    }
  } else if (mode == DequantMode::TFlite) {
    int64_t lshift_val = lshift();
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      int64_t multi = p.inputs[1][c * 3];
      int64_t shift_val = p.inputs[1][c * 3 + 1];
      int64_t zero_point = p.inputs[1][c * 3 + 2];
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          int64_t tmp = ((int32_t)p.inputs[0][offset] - zero_point) << lshift_val;
          p.outputs[0][offset] = MultiplyByQuantizedMultiplier(tmp, 1, -shift_val);
        }
      }
    }
  } else {
    llvm_unreachable("no such dequant mode");
  }
  return success();
}

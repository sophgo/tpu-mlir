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

LogicalResult tpu::RequantIntOp::init(InferenceParameter &p) { return success(); }
void tpu::RequantIntOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RequantIntOp::inference(InferenceParameter &p) {
  auto o_sType = Module::getStorageType(output());
  auto o_qtype = Quant::getUniformQuantizedType(output());
  int64_t num_elem = Module::getNumElements(input());
  auto mode = quant_mode();
  auto shape = Module::getShape(output());
  int64_t inner = 1;
  for (int i = 2; i < shape.size(); ++i) {
    inner *= shape[i];
  }
  int64_t zp_x = 0;
  if (Quant::isUniformQuantized(input())) {
    auto i_qtype = Quant::getUniformQuantizedType(input());
    zp_x = i_qtype.getZeroPoint();
    assert(mode == tpu::RequantMode::Normal);
  }
  int64_t shift_val = -rshift();
  int64_t multi = multiplier();
  int64_t zero_point = o_qtype.getZeroPoint();

  if (mode == tpu::RequantMode::TFlite_Lshift) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          auto v =
              zero_point + MultiplyByQuantizedMultiplier(
                               (int32_t)(p.inputs[0][offset]),
                               (int32_t)multi, (int32_t)shift_val);
          p.outputs[0][offset] = o_sType.isUnsignedInteger(8) ? Quant::to_uint8(v)
                                                              : Quant::to_int8(v);
        }
      }
    }
  } else if (mode == tpu::RequantMode::TFlite) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      assert(shift_val <= 0);
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          auto v =
              zero_point + MultiplyByQuantizedMultiplier(
                               (int32_t)(p.inputs[0][offset]),
                               (int32_t)multi, (int32_t)shift_val);
          p.outputs[0][offset] = o_sType.isUnsignedInteger(8) ? Quant::to_uint8(v)
                                                              : Quant::to_int8(v);
        }
      }
    }
  } else if (mode == tpu::RequantMode::Normal) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          auto v =
              zero_point + applyMultiplierAndRShift(
                               (p.inputs[0][offset] - zp_x), multi, -shift_val);
          p.outputs[0][offset] = o_sType.isUnsignedInteger(8) ? Quant::to_uint8(v)
                                                              : Quant::to_int8(v);
        }
      }
    }
  }
  return success();
}

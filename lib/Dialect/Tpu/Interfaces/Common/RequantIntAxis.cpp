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

LogicalResult tpu::RequantIntAxisOp::init(InferenceParameter &p) {
  return success();
}
void tpu::RequantIntAxisOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RequantIntAxisOp::inference(InferenceParameter &p) {
  auto i_sType = Module::getStorageType(input());
  auto o_sType = Module::getStorageType(output());
  auto o_qtype = Quant::getUniformQuantizedType(output());

  auto shape = Module::getShape(output());
  auto mode = quant_mode();
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
  if (mode == tpu::RequantMode::TFlite_Lshift) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      int64_t multi = p.inputs[1][c * 3];
      int64_t shift_val = p.inputs[1][c * 3 + 1];
      int64_t zero_point = p.inputs[1][c * 3 + 2];
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          int v = zero_point + MultiplyByQuantizedMultiplier(
                                   (int32_t)(p.inputs[0][offset]),
                                   (int32_t)multi, (int32_t)shift_val);
          p.outputs[0][offset] = saturate(v, o_sType);
        }
      }
    }
  } else if (mode == tpu::RequantMode::TFlite) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      int64_t multi = p.inputs[1][c * 3];
      int64_t shift_val = p.inputs[1][c * 3 + 1];
      int64_t zero_point = p.inputs[1][c * 3 + 2];
      assert(shift_val <= 0);
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          int v = zero_point + MultiplyByQuantizedMultiplier(
                                   (int32_t)(p.inputs[0][offset]),
                                   (int32_t)multi, (int32_t)shift_val);
          p.outputs[0][offset] = saturate(v, o_sType);
        }
      }
    }
  } else if (mode == tpu::RequantMode::Normal) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      int64_t multi, rshift_val, zero_point;
      if (Module::isBM1686()) {
        multi = p.inputs[1][c * 2];
        uint32_t tmp = p.inputs[1][c * 2 + 1];
        rshift_val = (int64_t)(-(char)(tmp & 0xff));
        zero_point = (int64_t)(short)((tmp & 0xffff0000) >> 16);
      } else {
        multi = p.inputs[1][c * 3];
        rshift_val = -p.inputs[1][c * 3 + 1];
        zero_point = p.inputs[1][c * 3 + 2];
      }
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          int v = zero_point +
                  applyMultiplierAndRShift((p.inputs[0][offset] - zp_x), multi,
                                           rshift_val);
          p.outputs[0][offset] = saturate(v, o_sType);
        }
      }
    }
  }
  return success();
}

mlir::Type tpu::RequantIntAxisOp::type_verify(uint64_t opd_idx,
                                              TypeCastMode &mode) {
  if (opd_idx == 0) {
    auto op = getOperation();
    auto stype = Module::getStorageType(input());
    if (stype.isIntOrIndex()) {
      return do_nothing(mode);
    }
    mode = TypeCastMode::DO_CAST;
    auto bitwith = stype.getIntOrFloatBitWidth();
    return Builder(op).getIntegerType(bitwith);
  }
  return do_nothing(mode);
}

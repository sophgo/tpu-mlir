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
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"



LogicalResult tpu::DequantIntOp::init(InferenceParameter &p) {
  return success();
}
void tpu::DequantIntOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::DequantIntOp::inference(InferenceParameter &p) {
  auto o_sType = module::getStorageType(output());
  auto qtype = module::getUniformQuantizedType(input());
  int64_t num_elem = module::getNumElements(input());
  int64_t shift_val = shift();
  int64_t mul_val = multiplier();
  int64_t offset = (int64_t)qtype.getZeroPoint();
  auto qmode = quant_mode();
  switch (qmode) {
  case DequantMode::Normal: {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t idx = 0; idx < num_elem; idx++) {
      int32_t tmp = (int32_t)p.inputs[0][idx] - offset;
      auto v = applyMultiplierAndRShift(tmp, mul_val, -shift_val);
      p.outputs[0][idx] = v;
    }
  } break;
  case DequantMode::TFlite: {
    int64_t lshift_val = lshift();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t idx = 0; idx < num_elem; idx++) {
      int64_t tmp = ((int32_t)p.inputs[0][idx] - offset) * mul_val
                    << lshift_val;
      auto v = RightShiftRound(tmp, 31, ROUNDING_HALF_UP);
      v = RightShiftRound(v, -shift_val, ROUNDING_HALF_AWAY_FROM_ZERO);
      p.outputs[0][idx] = v;
    }
  } break;
  default:
    llvm_unreachable("Unknown dequant mode");
    break;
  }
  return success();
}

mlir::Type tpu::DequantIntOp::type_verify(uint64_t opd_idx,
                                          TypeCastMode &mode) {
  if (opd_idx == 0) {
    auto op = getOperation();
    auto stype = module::getStorageType(input());
    if (stype.isIntOrIndex()) {
      return do_nothing(mode);
    }
    mode = TypeCastMode::DO_CAST;
    auto bitwith = stype.getIntOrFloatBitWidth();
    return Builder(op).getIntegerType(bitwith);
  }
  return do_nothing(mode);
}

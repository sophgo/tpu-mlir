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



LogicalResult tpu::RequantFpOp::init(InferenceParameter &p) {
  return success();
}
void tpu::RequantFpOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RequantFpOp::inference(InferenceParameter &p) {
  auto o_sType = module::getStorageType(getOutput());
  auto o_qtype = module::getUniformQuantizedType(getOutput());
  auto mode = getQuantMode();
  auto shape = module::getShape(getOutput());
  int64_t length = 1;
  for (int i = 0; i < shape.size(); ++i) {
    length *= shape[i];
  }

  float scale_v = getScale().convertToDouble();
  float offset_v = getOffset().convertToDouble();
  int64_t zero_point = o_qtype.getZeroPoint();

  switch (mode) {
  case RequantMode::TFlite:
  case RequantMode::TFlite_Lshift: {
#pragma omp parallel for schedule(static, omp_schedule(length))
    for (int64_t i = 0; i < length; ++i) {
      int32_t v = (int32_t)(round(p.inputs[0][i] * scale_v)) + zero_point;
      p.outputs[0][i] = saturate(v, o_sType);
    }
  } break;
  case RequantMode::Normal: {
#pragma omp parallel for schedule(static, omp_schedule(length))
    for (int64_t i = 0; i < length; ++i) {
      int32_t v =
          (int32_t)(round((float)(p.inputs[0][i]) * scale_v - offset_v));
      p.outputs[0][i] = saturate(v, o_sType);
    }
  } break;
  default:
    llvm_unreachable("Unknown requant mode");
    break;
  }
  return success();
}

mlir::Type tpu::RequantFpOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  if (opd_idx == 0) {
    auto op = getOperation();
    auto stype = module::getStorageType(getInput());
    if (stype.isIntOrIndex()) {
      return do_nothing(mode);
    }
    mode = TypeCastMode::DO_CAST;
    auto bitwith = stype.getIntOrFloatBitWidth();
    return Builder(op).getIntegerType(bitwith);
  }
  return do_nothing(mode);
}

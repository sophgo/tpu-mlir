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

LogicalResult tpu::RequantFpAxisOp::init(InferenceParameter &p) {
  return success();
}
void tpu::RequantFpAxisOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RequantFpAxisOp::inference(InferenceParameter &p) {
  auto i_sType = Module::getStorageType(input());
  auto o_sType = Module::getStorageType(output());
  auto o_qtype = Quant::getUniformQuantizedType(output());

  auto shape = Module::getShape(output());
  auto mode = quant_mode();
  int64_t inner = 1;
  for (int i = 2; i < shape.size(); ++i) {
    inner *= shape[i];
  }

  switch (mode) {
  case RequantMode::TFlite:
  case RequantMode::TFlite_Lshift: {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      float scale = p.inputs[1][c * 2];
      int64_t zero_point = p.inputs[1][c * 2 + 1];
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int index = (n * shape[1] + c) * inner + i;
          int32_t v = (int32_t)(round(p.inputs[0][index] * scale)) + zero_point;
          p.outputs[0][index] = saturate(v, o_sType);
        }
      }
    }
  } break;
  case RequantMode::Normal: {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      float scale = p.inputs[1][c * 2];
      float offset = p.inputs[1][c * 2 + 1];
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int index = (n * shape[1] + c) * inner + i;
          int32_t v =
              (int32_t)(round((float)(p.inputs[0][index]) * scale + offset));
          p.outputs[0][index] = saturate(v, o_sType);
        }
      }
    }
  } break;
  default:
    llvm_unreachable("Unknown requant mode");
    break;
  }
  return success();
}

mlir::Type tpu::RequantFpAxisOp::type_verify(uint64_t opd_idx,
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

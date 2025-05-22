//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

#include "tpu_mlir/Support/ActiveUtils.h"
#include "tpu_mlir/Support/CastUtils.h"
#include "tpu_mlir/Support/GenericCpuFunc.h"
#include "tpu_mlir/Support/LutFunc.h"

LogicalResult tpu::FusedActiveCastOp::init(InferenceParameter &p) {
  return success();
}
void tpu::FusedActiveCastOp::deinit(InferenceParameter &p) {}

static void active_func(InferenceParameter &p, int64_t num, activate_f func) {
#pragma omp parallel for schedule(static, omp_schedule(num))
  for (int i = 0; i < num; ++i) {
    p.outputs[0][i] = func(p.inputs[0][i]);
  }
}

LogicalResult tpu::FusedActiveCastOp::inference(InferenceParameter &p) {
  auto t = module::getStorageType(getOutput());
  auto num_element = module::getNumElements(getInput());
  active_func(p, num_element, getActivateFunc(*this));
  if (t.isBF16()) {
    BF16(p.outputs[0], p.outputs[0], num_element);
  } else if (t.isF16()) {
    F16(p.outputs[0], p.outputs[0], num_element);
  }
  bool isOutQuant = module::isUniformQuantized(getOutput());
  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());
  bool fInput = in_type.isIntOrIndex() == false;
  auto round_mode = round_mode_convert(getRoundMode());
  if (isOutQuant && fInput) {
    // FP32|BF16|F16|... => INT8|UINT8|...
    auto qtype = module::getUniformQuantizedType(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
    for (auto i = 0; i < num_element; i++) {
      float v = requant(p.outputs[0][i], qtype);
      p.outputs[0][i] = saturate(v, out_type, round_mode);
    }
  } else {
    UNREACHABLE_THIS("to be implemented");
  }
  return success();
}

bool tpu::FusedActiveCastOp::support_multi_core() { return false; }

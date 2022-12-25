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
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/LutFunc.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Float16.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

static mlir::Type t;

LogicalResult tpu::ActiveOp::init(InferenceParameter &p) { return success(); }
void tpu::ActiveOp::deinit(InferenceParameter &p) {}

static inline double hsigmoid(double x, double alpha, double beta) {
  return std::max(0.0, std::min(1.0, alpha * x + beta));
}

static inline double hswish(double x) {
  if (t.isBF16()) {
    return BF16(x * std::max(0.0f, std::min(1.0f, BF16(BF16(x + 3.0) / 6.0))));
  }
  if (t.isF16()) {
    return F16(x * std::max(0.0f, std::min(1.0f, F16(F16(x + 3.0) / 6.0))));
  }
  return x * std::max(0.0, std::min(1.0, (x + 3.0) / 6.0));
}

static void active_func(InferenceParameter &p, int64_t num, activate_f func) {
#pragma omp parallel for schedule(static, omp_schedule(num))
  for (int i = 0; i < num; ++i) {
    p.outputs[0][i] = func(p.inputs[0][i]);
  }
}

LogicalResult tpu::ActiveOp::inference(InferenceParameter &p) {
  t = Module::getStorageType(output());
  auto num_element = Module::getNumElements(input());
  switch (mode()) {
  case ActiveMode::ABSVAL:
    active_func(p, num_element, [](double val) { return std::abs(val); });
    break;
  case ActiveMode::ERF:
    active_func(p, num_element, [](double val) { return std::erf(val); });
    break;
  case ActiveMode::EXP:
    active_func(p, num_element, [](double val) { return std::exp(val); });
    break;
  case ActiveMode::LN:
    active_func(p, num_element, [](double val) { return std::log(val); });
    break;
  case ActiveMode::SQRT:
    active_func(p, num_element, [](double val) { return std::sqrt(val); });
    break;
  case ActiveMode::SILU:
    active_func(p, num_element,
                [](double val) { return val / (1 + std::exp(-val)); });
    break;
  case ActiveMode::SIGMOID:
    active_func(p, num_element,
                [](double val) { return 1 / (1 + std::exp(-val)); });
    break;
  case ActiveMode::HSIGMOID: {
    const auto coeffs_ = Module::getF64Array(coeffs(), 2, 0);
    const double alpha = coeffs_->at(1);
    const double beta = coeffs_->at(0);
    active_func(p, num_element, [alpha, beta](double val) {
      return hsigmoid(val, alpha, beta);
    });
    break;
  }
  case ActiveMode::HSWISH:
    active_func(p, num_element, [](double val) { return hswish(val); });
    break;
  }
  if (t.isBF16()) {
    f32_to_bf16(p.outputs[0], p.outputs[0], num_element, Module::isCV18xx());
  } else if (t.isF16()) {
    f32_to_f16(p.outputs[0], p.outputs[0], num_element);
  }
  return success();
}

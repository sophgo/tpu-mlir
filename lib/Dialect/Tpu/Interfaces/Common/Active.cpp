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

LogicalResult tpu::ActiveOp::init(InferenceParameter &p) { return success(); }
void tpu::ActiveOp::deinit(InferenceParameter &p) {}

static void active_func(InferenceParameter &p, int64_t num, activate_f func) {
#pragma omp parallel for schedule(static, omp_schedule(num))
  for (int i = 0; i < num; ++i) {
    p.outputs[0][i] = func(p.inputs[0][i]);
  }
}

LogicalResult tpu::ActiveOp::inference(InferenceParameter &p) {
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
  }
  return success();
}

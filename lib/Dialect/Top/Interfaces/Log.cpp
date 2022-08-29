//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

int64_t top::LogOp::getFLOPs() { return Module::getNumElements(output()) * 4; }

LogicalResult top::LogOp::init(InferenceParameter &p) { return success(); }
void top::LogOp::deinit(InferenceParameter &p) {}

LogicalResult top::LogOp::inference(InferenceParameter &p) {
  auto num_element = Module::getNumElements(input());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::log(val);
  }
  return success();
}

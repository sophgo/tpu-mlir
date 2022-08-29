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

int64_t top::ClipOp::getFLOPs() { return 0; }

LogicalResult top::ClipOp::init(InferenceParameter &p) { return success(); }
void top::ClipOp::deinit(InferenceParameter &p) {}

LogicalResult top::ClipOp::inference(InferenceParameter &p) {
  auto num_element = Module::getNumElements(output());
  auto min_v = static_cast<float>(minAttr().getValueAsDouble());
  auto max_v = static_cast<float>(maxAttr().getValueAsDouble());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::min(max_v, std::max(min_v, val));
  }
  return success();
}

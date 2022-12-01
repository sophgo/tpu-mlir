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

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::ClipOp::init(InferenceParameter &p) { return success(); }
void tpu::ClipOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ClipOp::inference(InferenceParameter &p) {
  auto min_v = static_cast<float>(minAttr().getValueAsDouble());
  auto max_v = static_cast<float>(maxAttr().getValueAsDouble());
  auto num_element = Module::getNumElements(output());
  assert(!Quant::isUniformQuantized(output()) && "Not Implemented");

#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::min(max_v, std::max(min_v, val));
  }
  return success();
}

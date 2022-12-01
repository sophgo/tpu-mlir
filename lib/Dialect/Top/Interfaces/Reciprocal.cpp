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

int64_t top::ReciprocalOp::getFLOPs() {
  return Module::getNumElements(output()) * (1 + do_relu() ? 1 : 0);
}

LogicalResult top::ReciprocalOp::init(InferenceParameter &p) {
  return success();
}
void top::ReciprocalOp::deinit(InferenceParameter &p) {}

LogicalResult top::ReciprocalOp::inference(InferenceParameter &p) {
  int64_t num_elem = Module::getNumElements(output());
  float const_s = const_val().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = const_s / p.inputs[0][i];
  }
  if (do_relu()) {
    auto limit = relu_limit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0], num_elem, limit);
  }
  return success();
}

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


int64_t top::AbsOp::getFLOPs() {
  return Module::getNumElements(output());
}
#if 0
LogicalResult top::AbsOp::init(InferenceParameter &p) {
  auto binary = new Binary();
  (*binary)
      .lhs(p.inputs[0], Module::getShape(inputs()[0]))
      .rhs(p.inputs[1], Module::getShape(inputs()[1]))
      .dst(p.outputs[0], Module::getShape(output()))
      .algorithem(algorithm::eltwise_abs)
      .setup();

  p.handle = (void *)binary;

  return success();
}
void top::AbsOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult top::AbsOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto binary = (Binary *)p.handle;
  binary->run();
  return success();
}
#endif
LogicalResult top::AbsOp::init(InferenceParameter &p) { return success(); }
void top::AbsOp::deinit(InferenceParameter &p) {}

LogicalResult top::AbsOp::inference(InferenceParameter &p) {
  auto num_element = Module::getNumElements(output());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::abs(val);
  }
  return success();
}

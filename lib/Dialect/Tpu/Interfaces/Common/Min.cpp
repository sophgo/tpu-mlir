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
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::MinOp::init(InferenceParameter &p) {
  auto binary = new Binary();
  (*binary)
      .lhs(p.inputs[0], Module::getShape(inputs()[0]))
      .rhs(p.inputs[1], Module::getShape(inputs()[1]))
      .dst(p.outputs[0], Module::getShape(output()))
      .algorithem(algorithm::binary_min)
      .setup();
  p.handle = (void *)binary;
  return success();
}

void tpu::MinOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult tpu::MinOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto binary = (Binary *)p.handle;
  binary->run();
  return success();
}

LogicalResult tpu::MinOp::LocalGenSupport() {
  // BackwardH and BackwardN can not handle more than one input right now.
  // The same n_slice and h_slice value will propagate to each inputs.
  // Thus, the local layer is only safe when we do not need to slice n and h
  // dimensions.
  auto out_shape = Module::getShape(output());
  auto lhs_shape = Module::getShape(inputs()[0]);
  auto rhs_shape = Module::getShape(inputs()[1]);
  if (getOperand(1).getDefiningOp() &&
      isa<top::WeightOp>(getOperand(1).getDefiningOp()))
    return failure();
  // left align
  switch (out_shape.size()) {
  case 2:
    if (lhs_shape[0] != rhs_shape[0])
      return failure();
  case 3:
  case 4:
    if (lhs_shape[0] != rhs_shape[0])
      return failure();
    if (lhs_shape[2] != rhs_shape[2])
      return failure();
  default:
    success();
  }
  return success();
}

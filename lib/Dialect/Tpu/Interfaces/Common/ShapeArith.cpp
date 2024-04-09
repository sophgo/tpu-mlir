//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/GenericCpuFunc.h"
#include <queue>
#include <vector>

LogicalResult tpu::ShapeArithOp::init(InferenceParameter &p) {
  auto binary = new Binary();
  auto lhs_shape = module::getShape(getInputs()[0]);
  auto rhs_shape = module::getShape(getInputs()[1]);
  auto type = getType();
  algorithm alg;
  if (type == "Add") {
    alg = algorithm::binary_add;
  } else if (type == "Sub") {
    alg = algorithm::binary_sub;
  } else if (type == "Mul") {
    alg = algorithm::binary_mul;
  } else {
    return failure();
  }

  (*binary)
      .hs(p.inputs[0], p.inputs[1], lhs_shape, rhs_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .algorithem(alg)
      .setup();
  p.handle = (void *)binary;
  return success();
}
void tpu::ShapeArithOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult tpu::ShapeArithOp::inference(InferenceParameter &p) {
  auto binary = (Binary *)p.handle;
  binary->run();

  return success();
}

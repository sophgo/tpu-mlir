//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::C2CAllReduceOp::getFLOPs() { return 0; }

LogicalResult top::C2CAllReduceOp::init(InferenceParameter &p) {
  return success();
}

void top::C2CAllReduceOp::deinit(InferenceParameter &p) {}

LogicalResult top::C2CAllReduceOp::inference(InferenceParameter &p) {
  // Multi-chip all-reduce cannot be simulated on host without MPI topology.
  auto num = module::getNumElements(getSend());
  memcpy(p.outputs[0], p.inputs[0], num * sizeof(float));
  return success();
}

void top::C2CAllReduceOp::shape_inference() {
  auto send_shape = module::getShape(getSend());
  auto recv_shape = module::getShape(getRecv());
  ASSERT_THIS(send_shape == recv_shape);
  module::setShapeOrVerify(getOutput(), send_shape);
}

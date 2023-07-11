//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ListOp::getFLOPs() { return 0; }

LogicalResult top::ListOp::init(InferenceParameter &p) { return success(); }

void top::ListOp::deinit(InferenceParameter &p) {}

LogicalResult top::ListOp::inference(InferenceParameter &p) {
  int64_t offset = 0;
  int64_t num_inputs = getInputs().size();
  for (int i = 0; i < num_inputs; i++) {
    if (module::isNone(getInputs()[i])) {
      continue;
    }
    auto num = module::getNumElements(getInputs()[i]);
    memcpy(p.outputs[0] + offset, p.inputs[i], num * sizeof(float));
    offset += num;
  }
  return success();
}

// ListOp is special, will convert to WeightOp
void top::ListOp::shape_inference() {
  int64_t num_outputs = 0;
  for (auto in : getInputs()) {
    if (module::isNone(in)) {
      continue;
    }
    num_outputs += module::getNumElements(in);
  }
  std::vector<int64_t> new_shape = {num_outputs};
  module::setShapeOrVerify(getOutput(), new_shape);
}

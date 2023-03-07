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
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::AddOp::getFLOPs() {
  return module::getNumElements(getOutput()) *
         (getInputs().size() - 1 + getDoRelu() ? 1 : 0);
}

LogicalResult top::AddOp::init(InferenceParameter &p) {
  if (getInputs().size() == 2) {
    auto binary = new Binary();
    auto lhs_shape = module::getShape(getInputs()[0]);
    auto rhs_shape = module::getShape(getInputs()[1]);

    (*binary)
        .hs(p.inputs[0], p.inputs[1], lhs_shape, rhs_shape)
        .dst(p.outputs[0], module::getShape(getOutput()))
        .do_relu(getDoRelu())
        .relu_limit(getReluLimit().convertToDouble())
        .algorithem(algorithm::binary_add)
        .setup();

    p.handle = (void *)binary;
  } else {
    p.handle = nullptr;
  }
  return success();
}
void top::AddOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult top::AddOp::inference(InferenceParameter &p) {
  if (getInputs().size() == 2) {
    if (p.handle == nullptr) {
      return failure();
    }
    auto binary = (Binary *)p.handle;
    binary->run();
  } else {
    auto num_elem = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = 0;
      for (auto in : p.inputs) {
        if (in != nullptr) {
          p.outputs[0][i] += in[i];
        }
      }
    }
    if (getDoRelu()) {
      auto limit = getReluLimit().convertToDouble();
      function_relu(p.outputs[0], p.outputs[0], num_elem, limit);
    }
  }

  return success();
}

void top::AddOp::shape_inference() {
  broadcast_shape_inference(getOperation());
}

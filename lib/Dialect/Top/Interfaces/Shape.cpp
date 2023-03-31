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
#include "tpu_mlir/Support/Module.h"

int64_t top::ShapeOp::getFLOPs() { return 0; }

LogicalResult top::ShapeOp::init(InferenceParameter &p) { return success(); }

void top::ShapeOp::deinit(InferenceParameter &p) {}

LogicalResult top::ShapeOp::inference(InferenceParameter &p) {
  float *output_data = p.outputs[0];
  auto input_shape = module::getShape(getInput());
  for (int i = 0; i < input_shape.size(); ++i) {
    output_data[i] = input_shape[i];
  }
  return success();
}

void top::ShapeOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  std::vector<int64_t> output_shape({(int64_t)input_shape.size()});
  module::setShapeOrVerify(getOutput(), output_shape);
}

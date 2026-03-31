//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::CastOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::CastOp::init(InferenceParameter &p) { return success(); }
void top::CastOp::deinit(InferenceParameter &p) {}

LogicalResult top::CastOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  auto num_elem = module::getNumElements(getOutput());
  // auto in_type = module::getStorageType(getInput());
  // auto out_type = module::getStorageType(getOutput());
  auto round_mode = round_mode_convert(getRoundMode().str());
  auto to = getTo();
  if (to == "INT32") {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = to_int<float>(p.inputs[0][i], round_mode);
    }
    return success();
  } else if (to == "F32") {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = p.inputs[0][i];
    }
    return success();
  } else {
    UNREACHABLE_THIS("Not Implemented");
    return failure();
  }
}

void top::CastOp::shape_inference() {
  common_shape_inference(getOperation());
  auto output_shape = module::getShape(getOutput());
  if (module::isShape(getInput())) {
    auto input_v = module::getShapeTensorValue(getInput());
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), {input_v}, output_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}

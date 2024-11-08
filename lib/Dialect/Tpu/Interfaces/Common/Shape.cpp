//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::ShapeOp::init(InferenceParameter &p) { return success(); }
void tpu::ShapeOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ShapeOp::inference(InferenceParameter &p) {
  float *output_data = p.outputs[0];
  auto input_shape = module::getShape(getInput());
  for (int i = 0; i < input_shape.size(); ++i) {
    output_data[i] = input_shape[i];
  }
  std::vector<int64_t> output_shape({(int64_t)input_shape.size()});
  module::setShape(getOutput(), output_shape);
  return success();
}

mlir::Type tpu::ShapeOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return do_nothing(mode);
}

bool tpu::ShapeOp::support_multi_core() { return false; }

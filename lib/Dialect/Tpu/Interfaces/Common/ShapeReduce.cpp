//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <valarray>

using namespace tpu_mlir::backend;

LogicalResult tpu::ShapeReduceOp::init(InferenceParameter &p) {
  return success();
}
void tpu::ShapeReduceOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ShapeReduceOp::inference(InferenceParameter &p) {
  float *input_v = p.inputs[0];
  float *output_v = p.outputs[0];
  auto type_val = getMode();
  auto axes = getAxes();
  auto keep_dims = getKeepdims();
  auto input_shape = module::getShape(getInput());
  ASSERT_THIS(axes.size() == 1 && keep_dims && input_shape.size() == 1 &&
              "ShapeReduce only support one dim and keep_dims now");
  if (type_val == "ReduceMax") {
    output_v[0] = *input_v;
    for (int index = 0; index < input_shape[0]; index++)
      output_v[0] = std::max(output_v[0], *(input_v + index));
  } else if (type_val == "ReduceMin") {
    output_v[0] = *input_v;
    for (int index = 0; index < input_shape[0]; index++)
      output_v[0] = std::min(output_v[0], *(input_v + index));
  } else {
    UNREACHABLE_THIS("Not Implemented ShapeReduce type");
  }
  return success();
}

mlir::Type tpu::ShapeReduceOp::type_verify(uint64_t opd_idx,
                                           TypeCastMode &mode) {
  return do_nothing(mode);
}

bool tpu::ShapeReduceOp::support_multi_core() { return false; }

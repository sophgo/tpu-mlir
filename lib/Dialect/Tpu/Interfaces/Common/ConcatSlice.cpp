//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::ConcatSliceOp::init(InferenceParameter &p) {
  return success();
}

void tpu::ConcatSliceOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ConcatSliceOp::inference(InferenceParameter &p) {
  auto in0_shape = module::getShape(getIn0());
  auto in1_shape = module::getShape(getIn1());
  int64_t axis = getAxis();

  int64_t outer_size = 1;
  for (int64_t i = 0; i < axis; i++)
    outer_size *= in0_shape[i];

  int64_t inner_size = 1;
  for (size_t i = axis + 1; i < in0_shape.size(); i++)
    inner_size *= in0_shape[i];

  int64_t axis_size_0 = in0_shape[axis];
  int64_t axis_size_1 = in1_shape[axis];
  int64_t keep = axis_size_0 - axis_size_1;

  float *in0 = p.inputs[0];
  float *in1 = p.inputs[1];
  float *out = p.outputs[0];

  for (int64_t o = 0; o < outer_size; o++) {
    // Copy tail of in0
    if (keep > 0) {
      memcpy(out + o * axis_size_0 * inner_size,
             in0 + o * axis_size_0 * inner_size + axis_size_1 * inner_size,
             keep * inner_size * sizeof(float));
    }
    // Copy all of in1
    memcpy(out + o * axis_size_0 * inner_size + keep * inner_size,
           in1 + o * axis_size_1 * inner_size,
           axis_size_1 * inner_size * sizeof(float));
  }

  auto out_type = module::getStorageType(getOutput());
  auto num_elem = module::getNumElements(getOutput());
  if (out_type.isBF16()) {
    BF16(p.outputs[0], p.outputs[0], num_elem);
  } else if (out_type.isF16()) {
    F16(p.outputs[0], p.outputs[0], num_elem);
  }

  return success();
}

bool tpu::ConcatSliceOp::support_multi_core() { return true; }

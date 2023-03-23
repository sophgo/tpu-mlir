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
#include "tpu_mlir/Support/MathUtils.h"

int64_t top::SliceAxisOp::getFLOPs() { return 0; }

LogicalResult top::SliceAxisOp::init(InferenceParameter &p) {
  return success();
}
void top::SliceAxisOp::deinit(InferenceParameter &p) {}

LogicalResult top::SliceAxisOp::inference(InferenceParameter &p) {
  // auto out_num_elem = module::getNumElements(getOutput());
  auto axis = getAxis();
  auto start = getStart();
  auto end = getEnd();
  auto step = getStep();
  auto in_shape = module::getShape(getInput());
  auto outer_size = std::accumulate(in_shape.begin(), in_shape.begin() + axis,
                                    1, std::multiplies<int64_t>());
  auto inner_size = std::accumulate(in_shape.begin() + axis + 1, in_shape.end(),
                                    1, std::multiplies<int64_t>());
  auto out = p.outputs[0];
  for (int i = 0; i < outer_size; ++i) {
    for (int j = start; j < end; j += step) {
      int64_t offset = (i * in_shape[axis] + j) * inner_size;
      memcpy(out, p.inputs[0] + offset, inner_size * sizeof(float));
      out += inner_size;
    }
  }

  return success();
}

void top::SliceAxisOp::shape_inference() {
  auto axis = getAxis();
  auto start = getStart();
  auto end = getEnd();
  auto step = getStep();
  auto in_shape = module::getShape(getInput());
  if (axis < 0) {
    axis += in_shape.size();
    setAxis(axis);
  }
  if (start < 0) {
    start += in_shape[axis];
    setStart(start);
  }
  if (end < 0) {
    end += in_shape[axis];
    setEnd(end);
  } else if (end > in_shape[axis]) {
    end = in_shape[axis];
    setEnd(end);
  }
  std::vector<int64_t> out_shape(in_shape);
  out_shape[axis] = (end - start + step - 1) / step;
  module::setShapeOrVerify(getOutput(), out_shape);
}

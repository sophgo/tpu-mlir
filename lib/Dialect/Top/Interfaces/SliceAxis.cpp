//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::SliceAxisOp::getFLOPs() { return 0; }

LogicalResult top::SliceAxisOp::init(InferenceParameter &p) {
  return success();
}
void top::SliceAxisOp::deinit(InferenceParameter &p) {}

LogicalResult top::SliceAxisOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  auto dims = in_shape.size();
  auto axis_op = getAxis().getDefiningOp<top::WeightOp>();
  int axis = axis_op.read<float>()->at(0);
  if (axis < 0)
    axis += dims;
  auto start_op = getStart().getDefiningOp<top::WeightOp>();
  int start = start_op.read<float>()->at(0);
  if (start < 0)
    start += in_shape[axis];
  auto end_op = getEnd().getDefiningOp<top::WeightOp>();
  auto end = end_op.read<float>()->at(0);
  if (end < 0)
    end += in_shape[axis];
  if (end > in_shape[axis])
    end = in_shape[axis];
  auto step_op = getStep().getDefiningOp<top::WeightOp>();
  int step = step_op.read<float>()->at(0);

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
  float start = 0;
  float step = 1;
  float end;
  auto in_shape = module::getShape(getInput());
  assert(module::isWeight(getAxis()));
  auto axis_op = getAxis().getDefiningOp<top::WeightOp>();
  auto axis_data = axis_op.read<float>();
  auto axis = axis_data->at(0);
  if (module::isNone(getEnd()) == false) {
    if (module::isWeight(getEnd())) {
      auto end_op = getEnd().getDefiningOp<top::WeightOp>();
      auto end_data = end_op.read<float>();
      end = end_data->at(0);
    } else {
      end = in_shape[(int)axis];
    }
  }
  if (module::isNone(getStart()) == false) {
    if (module::isWeight(getStart())) {
      auto start_op = getStart().getDefiningOp<top::WeightOp>();
      auto start_data = start_op.read<float>();
      start = start_data->at(0);
    } else {
      start = 0;
    }
  }
  if (module::isNone(getStep()) == false) {
    assert(module::isWeight(getStep()));
    auto step_op = getStep().getDefiningOp<top::WeightOp>();
    auto step_data = step_op.read<float>();
    step = step_data->at(0);
    assert(step != 0);
  }
  auto dims = in_shape.size();
  if (axis < 0) {
    axis += dims;
  }
  if (start < 0) {
    start += in_shape[axis];
  }
  if (end < 0) {
    end += in_shape[axis];
  } else if (end > in_shape[axis]) {
    end = in_shape[axis];
  }
  std::vector<int64_t> out_shape(in_shape);
  out_shape[axis] = (end - start + step - 1) / step;
  module::setShapeOrVerify(getOutput(), out_shape);
}

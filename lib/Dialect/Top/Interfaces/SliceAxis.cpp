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
  auto in_shape = module::getShape(getInput());
  ASSERT_THIS(module::isWeight(getAxis()));
  auto axis_op = getAxis().getDefiningOp<top::WeightOp>();
  auto axis_data = axis_op.read<float>();
  std::vector<int64_t> axes(axis_data->begin(), axis_data->end());
  std::vector<int64_t> starts(axes.size(), 0);
  std::vector<int64_t> steps(axes.size(), 1);
  std::vector<int64_t> ends(axes.size());
  for (size_t i = 0; i < axes.size(); ++i) {
    auto axis = axes[i];
    if (axis < 0) {
      axis += in_shape.size();
    }
    ends[i] = in_shape[axis];
  }
  if (module::isNone(getEnd()) == false) {
    if (module::isWeight(getEnd())) {
      auto end_op = getEnd().getDefiningOp<top::WeightOp>();
      auto end_data = end_op.read<float>();
      ends.assign(end_data->begin(), end_data->end());
    } else if (module::isShape(getEnd())) {
      auto end_v = module::getShapeTensorValue(getEnd());
      ASSERT_THIS(end_v.size() == axes.size());
      ends.assign(end_v.begin(), end_v.end());
    }
  }
  if (module::isNone(getStart()) == false) {
    if (module::isWeight(getStart())) {
      auto start_op = getStart().getDefiningOp<top::WeightOp>();
      auto start_data = start_op.read<float>();
      starts.assign(start_data->begin(), start_data->end());
    } else if (module::isShape(getStart())) {
      auto start_v = module::getShapeTensorValue(getStart());
      ASSERT_THIS(start_v.size() == axes.size());
      starts.assign(start_v.begin(), start_v.end());
    }
  }
  if (module::isNone(getStep()) == false) {
    ASSERT_THIS(module::isWeight(getStep()));
    auto step_op = getStep().getDefiningOp<top::WeightOp>();
    auto step_data = step_op.read<float>();
    steps.assign(step_data->begin(), step_data->end());
    ASSERT_THIS(std::find(steps.begin(), steps.end(), 0) == steps.end());
  }
  auto dims = in_shape.size();
  std::vector<int64_t> out_shape(in_shape);
  for (size_t i = 0; i < axes.size(); ++i) {
    int axis = axes[i];
    if (axis < 0) {
      axis += dims;
    }
    if (starts[i] < 0) {
      starts[i] += in_shape[axis];
    }
    if (ends[i] < 0) {
      ends[i] += in_shape[axis];
    }
    starts[i] = steps[i] > 0 ? std::clamp(starts[i], 0L, in_shape[axis])
                             : std::clamp(starts[i], 0L, in_shape[axis] - 1);
    ends[i] = steps[i] > 0 ? std::clamp(ends[i], 0L, in_shape[axis])
                           : std::clamp(ends[i], -1L, in_shape[axis] - 1);
    out_shape[axis] = abs_ceiling_func(ends[i] - starts[i], steps[i]);
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}

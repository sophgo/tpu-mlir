//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::WhereOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::WhereOp::init(InferenceParameter &p) { return success(); }
void top::WhereOp::deinit(InferenceParameter &p) {}

LogicalResult top::WhereOp::inference(InferenceParameter &p) {
  const auto num_element = module::getNumElements(getOutput());
  auto out_shape = module::getShape(getOutput());
  auto out_dim = out_shape.size();
  auto in0_shape = shape_expand_dim(module::getShape(getCond()), out_dim);
  const auto cond_element = module::getNumElements(getCond());
  auto x_const = getXIsConst();
  auto y_const = getYIsConst();

  std::vector<int64_t> in0_stride, in1_stride, in2_stride;
  if (x_const && y_const == false) {
    auto const_val = getXConstVal().convertToDouble();
    auto in1_shape = shape_expand_dim(module::getShape(getFbrn()), out_dim);
    const auto fbrn_element = module::getNumElements(getFbrn());
    if (num_element == cond_element && num_element == fbrn_element) {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t i = 0; i < num_element; ++i) {
        p.outputs[0][i] = p.inputs[0][i] ? const_val : p.inputs[2][i];
      }
    } else {
      get_stride(in0_shape, in0_stride);
      get_stride(in1_shape, in1_stride);
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t i = 0; i < num_element; ++i) {
        std::vector<int64_t> list_(out_dim);
        idx_to_list(i, out_shape, list_);
        int64_t cond_idx = list_to_idx(list_, in0_stride);
        int64_t tbrn_idx = list_to_idx(list_, in1_stride);
        p.outputs[0][i] =
            p.inputs[0][cond_idx] ? const_val : p.inputs[2][tbrn_idx];
      }
    }
  } else if (y_const && x_const == false) {
    auto const_val = getYConstVal().convertToDouble();
    auto in1_shape = shape_expand_dim(module::getShape(getTbrn()), out_dim);
    const auto tbrn_element = module::getNumElements(getTbrn());
    if (num_element == cond_element && num_element == tbrn_element) {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t i = 0; i < num_element; ++i) {
        p.outputs[0][i] = p.inputs[0][i] ? p.inputs[1][i] : const_val;
      }
    } else {
      get_stride(in0_shape, in0_stride);
      get_stride(in1_shape, in1_stride);
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t i = 0; i < num_element; ++i) {
        std::vector<int64_t> list_(out_dim);
        idx_to_list(i, out_shape, list_);
        int64_t cond_idx = list_to_idx(list_, in0_stride);
        int64_t fbrn_idx = list_to_idx(list_, in1_stride);
        p.outputs[0][i] =
            p.inputs[0][cond_idx] ? p.inputs[1][fbrn_idx] : const_val;
      }
    }
  } else if (y_const && x_const) {
    auto x_const_val_ = getXConstVal().convertToDouble();
    auto y_const_val_ = getYConstVal().convertToDouble();
    if (num_element == cond_element) {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t i = 0; i < num_element; ++i) {
        p.outputs[0][i] = p.inputs[0][i] ? x_const_val_ : y_const_val_;
      }
    } else {
      get_stride(in0_shape, in0_stride);
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t i = 0; i < num_element; ++i) {
        std::vector<int64_t> list_(out_dim);
        idx_to_list(i, out_shape, list_);
        int64_t cond_idx = list_to_idx(list_, in0_stride);
        p.outputs[0][i] = p.inputs[0][cond_idx] ? x_const_val_ : y_const_val_;
      }
    }
  } else {
    auto in1_shape = shape_expand_dim(module::getShape(getTbrn()), out_dim);
    auto in2_shape = shape_expand_dim(module::getShape(getFbrn()), out_dim);
    const auto tbrn_element = module::getNumElements(getTbrn());
    const auto fbrn_element = module::getNumElements(getFbrn());
    if (num_element == cond_element && num_element == tbrn_element &&
        num_element == fbrn_element) {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t i = 0; i < num_element; ++i) {
        p.outputs[0][i] = p.inputs[0][i] ? p.inputs[1][i] : p.inputs[2][i];
      }
    } else {
      get_stride(in0_shape, in0_stride);
      get_stride(in1_shape, in1_stride);
      get_stride(in2_shape, in2_stride);
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t i = 0; i < num_element; ++i) {
        std::vector<int64_t> list_(out_dim);
        idx_to_list(i, out_shape, list_);
        int64_t cond_idx = list_to_idx(list_, in0_stride);
        int64_t tbrn_idx = list_to_idx(list_, in1_stride);
        int64_t fbrn_idx = list_to_idx(list_, in2_stride);
        p.outputs[0][i] = p.inputs[0][cond_idx] ? p.inputs[1][tbrn_idx]
                                                : p.inputs[2][fbrn_idx];
      }
    }
  }
  return success();
}

void top::WhereOp::shape_inference() {
  broadcast_shape_inference(getOperation());
  // support case input/output both shape.
  // cond/x/y all weight/shape, and weight is integer
  std::vector<std::vector<int64_t>> input_shapes_v;
  if (module::isShape(getCond())) {
    auto input_shape_v = module::getShapeTensorValue(getCond());
    input_shapes_v.push_back(input_shape_v);
  } else if (module::isWeight(getCond())) {
    auto data = getCond().getDefiningOp<top::WeightOp>().read_as_float();
    std::vector<int64_t> data_v(data->begin(), data->end());
    input_shapes_v.push_back(data_v);
  }
  if (getXIsConst()) {
    auto x_const_v = getXConstVal().convertToDouble();
    if (x_const_v == floor(x_const_v)) {
      input_shapes_v.push_back({static_cast<int>(x_const_v)});
    }
  } else if (module::isShape(getTbrn())) {
    auto input_shape_v = module::getShapeTensorValue(getTbrn());
    input_shapes_v.push_back(input_shape_v);
  } else if (module::isWeight(getTbrn())) {
    auto data = getTbrn().getDefiningOp<top::WeightOp>().read_as_float();
    if (std::all_of(data->begin(), data->end(),
                    [](auto &x) { return x == floor(x); })) {
      std::vector<int64_t> data_v(data->begin(), data->end());
      input_shapes_v.push_back(data_v);
    }
  }
  if (getYIsConst()) {
    auto x_const_v = getYConstVal().convertToDouble();
    if (x_const_v == floor(x_const_v)) {
      input_shapes_v.push_back({static_cast<int>(x_const_v)});
    }
  } else if (module::isShape(getFbrn())) {
    auto input_shape_v = module::getShapeTensorValue(getFbrn());
    input_shapes_v.push_back(input_shape_v);
  } else if (module::isWeight(getFbrn())) {
    auto data = getFbrn().getDefiningOp<top::WeightOp>().read_as_float();
    if (std::all_of(data->begin(), data->end(),
                    [](auto &x) { return x == floor(x); })) {
      std::vector<int64_t> data_v(data->begin(), data->end());
      input_shapes_v.push_back(data_v);
    }
  }
  if (input_shapes_v.size() == 3) {
    auto out_shape = module::getShape(getOutput());
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), input_shapes_v, out_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::GatherOp::getFLOPs() { return 0; }

LogicalResult top::GatherOp::init(InferenceParameter &p) { return success(); }
void top::GatherOp::deinit(InferenceParameter &p) {}

LogicalResult top::GatherOp::inference(InferenceParameter &p) {
  const float *src = p.inputs[0];
  const float *inds = p.inputs[1];
  float *dst = p.outputs[0];
  auto num_indices = module::getNumElements(getIndices());
  auto ax = getAxis();
  int64_t outer_dims = 1;
  int64_t inner_dims = 1;
  auto input_shape = module::getShape(getInput());
  auto indices_shape = module::getShape(getIndices());
  if (ax < 0) {
    ax += input_shape.size();
  }
  for (int i = 0; i < ax; ++i) {
    outer_dims *= input_shape[i];
  }
  for (int i = ax + 1; i < input_shape.size(); ++i) {
    inner_dims *= input_shape[i];
  }

  auto num_elems = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_elems))
  for (int64_t i = 0; i < outer_dims; ++i) {
    for (int64_t j = 0; j < num_indices; ++j) {
      for (int64_t k = 0; k < inner_dims; ++k) {
        int64_t src_idx =
            (i * input_shape[ax] +
             (int64_t)(inds[j] < 0 ? inds[j] + input_shape[ax] : inds[j])) *
                inner_dims +
            k;
        int64_t dst_idx = (i * num_indices + j) * inner_dims + k;
        dst[dst_idx] = src[src_idx];
      }
    }
  }
  std::vector<int64_t> out_shape;
  for (int i = 0; i < ax; ++i) {
    out_shape.push_back(input_shape[i]);
  }

  if (indices_shape.size() == 1 && indices_shape[0] == 1 && !getKeepdims()) {
    // if indices_shape.size() == 1 and indices is scalar(not a array) do
    // squeeze manner do nothing
    if (input_shape.size() == 1) {
      // if input_shape.size() == 1, output_shape should be scalar represent by
      // 1D tensor
      out_shape.push_back({1});
      auto context = getContext();
      mlir::Builder builder(context);
      setIsScalarAttr(builder.getBoolAttr(true));
    }
  } else {
    for (int s : indices_shape) {
      out_shape.push_back(s);
    }
  }
  for (int i = ax + 1; i < input_shape.size(); ++i) {
    out_shape.push_back(input_shape[i]);
  }
  if (out_shape.size() == input_shape.size()) {
    auto builder = OpBuilder(getContext());
    setKeepdimsAttr(builder.getBoolAttr(true));
  }
  module::setShape(getOutput(), out_shape);
  return success();
}

void top::GatherOp::shape_inference() {
  auto indices_shape = module::getShape(getIndices());
  auto ax = getAxis();
  auto input_shape = module::getShape(getInput());
  if (ax < 0) {
    ax += input_shape.size();
    setAxis(ax);
  }
  std::vector<int64_t> out_shape;
  for (int i = 0; i < ax; ++i) {
    out_shape.push_back(input_shape[i]);
  }

  if (indices_shape.size() == 1 && indices_shape[0] == 1 && !getKeepdims()) {
    // if indices_shape.size() == 1 and indices is scalar(not a array) do
    // squeeze manner do nothing
    if (input_shape.size() == 1) {
      // if input_shape.size() == 1, output_shape should be scalar represent by
      // 1D tensor
      out_shape.push_back({1});
      auto context = getContext();
      mlir::Builder builder(context);
      setIsScalarAttr(builder.getBoolAttr(true));
    }
  } else {
    for (int s : indices_shape) {
      out_shape.push_back(s);
    }
  }
  for (int i = ax + 1; i < input_shape.size(); ++i) {
    out_shape.push_back(input_shape[i]);
  }
  if (out_shape.size() == input_shape.size()) {
    auto builder = OpBuilder(getContext());
    setKeepdimsAttr(builder.getBoolAttr(true));
  }
  module::setShapeOrVerify(getOutput(), out_shape);
  if (module::isShape(getInput())) {
    if (module::isWeight(getIndices())) {
      auto indices_w = dyn_cast<top::WeightOp>(getIndices().getDefiningOp());
      auto indices_float_val = indices_w.read_as_float();
      std::vector<int64_t> indices_val(indices_float_val->size());
      std::transform(indices_float_val->begin(), indices_float_val->end(),
                     indices_val.begin(),
                     [](auto &i) { return static_cast<int64_t>(i); });
      auto out_shape_val = module::commonShapeValInfer(
          getOperation(),
          {module::getShapeTensorValue(getInput()), indices_val}, out_shape);
      module::bindShapeTensorValue(getOutput(), out_shape_val);
    } else {
      UNREACHABLE_THIS("not implemented");
    }
  }
}

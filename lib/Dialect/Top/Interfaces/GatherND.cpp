//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"

using namespace std;

int64_t top::GatherNDOp::getFLOPs() { return 0; }

LogicalResult top::GatherNDOp::init(InferenceParameter &p) { return success(); }

void top::GatherNDOp::deinit(InferenceParameter &p) {}

LogicalResult top::GatherNDOp::inference(InferenceParameter &p) {
  GatherNDParam param;
  param.batch_dims = getBatchDims();
  tensor_list_t input;
  input.ptr = p.inputs[0];
  input.shape = module::getShape(getInput());
  input.size = module::getNumElements(getInput());
  param.inputs.push_back(input);

  tensor_list_t indices;
  indices.ptr = p.inputs[1];
  indices.shape = module::getShape(getIndices());
  indices.size = module::getNumElements(getIndices());
  param.inputs.push_back(indices);

  std::vector<int64_t> output_shape;
  for (int i = 0; i < param.batch_dims; ++i) {
    output_shape.push_back(indices.shape[i]);
  }
  for (int i = param.batch_dims; i < indices.shape.size() - 1; ++i) {
    output_shape.push_back(indices.shape[i]);
  }
  if (indices.shape[indices.shape.size() - 1] !=
      input.shape.size() - param.batch_dims) {
    for (int i = param.batch_dims + indices.shape[indices.shape.size() - 1];
         i < input.shape.size(); ++i) {
      output_shape.push_back(input.shape[i]);
    }
  }
  module::setShape(getOutput(), output_shape);

  tensor_list_t output;
  output.ptr = p.outputs[0];
  output.shape = module::getShape(getOutput());
  output.size = module::getNumElements(getOutput());
  param.output = output;

  GatherndFunc func(param);
  func.invoke();
  return success();
}

void top::GatherNDOp::shape_inference() {
  auto batch_dims = getBatchDims();
  auto data_rank = module::getShape(getInput()).size();
  auto indices_shape = module::getShape(getIndices());
  auto input_shape = module::getShape(getInput());
  std::vector<int64_t> output_shape;
  for (int i = 0; i < batch_dims; ++i) {
    output_shape.push_back(indices_shape[i]);
  }
  for (int i = batch_dims; i < indices_shape.size() - 1; ++i) {
    output_shape.push_back(indices_shape[i]);
  }
  if (indices_shape[indices_shape.size() - 1] != data_rank - batch_dims) {
    for (int i = batch_dims + indices_shape[indices_shape.size() - 1];
         i < input_shape.size(); ++i) {
      output_shape.push_back(input_shape[i]);
    }
  }
  module::setShapeOrVerify(getOutput(), output_shape);
}

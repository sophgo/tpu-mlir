//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"

int64_t top::ScatterNDOp::getFLOPs() { return 0; }

LogicalResult top::ScatterNDOp::init(InferenceParameter &p) {
  return success();
}
void top::ScatterNDOp::deinit(InferenceParameter &p) {}

LogicalResult top::ScatterNDOp::inference(InferenceParameter &p) {
  ScatterNDParam param;
  param.op_code = (CPU_SCATTER_OP_T)getReduction();
  tensor_list_t input;
  input.ptr = p.inputs[0];
  input.shape = module::getShape(getInputData());
  input.size = module::getNumElements(getInputData());
  param.inputs.push_back(input);

  tensor_list_t indices;
  indices.ptr = p.inputs[1];
  indices.shape = module::getShape(getIndices());
  indices.size = module::getNumElements(getIndices());
  param.inputs.push_back(indices);

  tensor_list_t updates;
  updates.ptr = p.inputs[2];
  updates.shape = module::getShape(getUpdates());
  updates.size = module::getNumElements(getUpdates());
  param.inputs.push_back(updates);

  module::setShape(getOutput(), input.shape);

  tensor_list_t output;
  output.ptr = p.outputs[0];
  output.shape = module::getShape(getOutput());
  output.size = module::getNumElements(getOutput());
  param.output = output;

  ScatterNDFunc func(param);
  func.invoke();

  return success();
}

void top::ScatterNDOp::shape_inference() {
  common_shape_inference(getOperation());
}

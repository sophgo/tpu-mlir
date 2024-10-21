//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"

LogicalResult tpu::GridSamplerOp::init(InferenceParameter &p) {
  return success();
}

void tpu::GridSamplerOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::GridSamplerOp::inference(InferenceParameter &p) {
  GridSamplerParam param;
  std::vector<tensor_list_t> input_list;
  param.mode = getMode();
  param.align_corners = getAlignCorners();
  param.padding_mode = getPaddingMode();

  tensor_list_t input;
  tensor_list_t grid;
  input.ptr = p.inputs[0];
  input.size = module::getNumElements(getInput());
  input.shape = module::getShape(getInput());
  grid.ptr = p.inputs[1];
  grid.size = module::getNumElements(getGrid());
  grid.shape = module::getShape(getGrid());

  input_list.push_back(input);
  input_list.push_back(grid);
  param.inputs = input_list;

  tensor_list_t output_tensor;
  output_tensor.size = module::getNumElements(getOutput());
  output_tensor.shape = module::getShape(getOutput());
  output_tensor.ptr = p.outputs[0];
  param.output = output_tensor;

  GridSamplerFunc func(param);
  func.invoke();
  return success();
}

bool tpu::GridSamplerOp::support_multi_core() { return module::isSG2380(); }

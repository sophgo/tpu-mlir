//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/GenericCpuFunc.h"
#include "tpu_mlir/Support/MathUtils.h"



int64_t top::GridSamplerOp::getFLOPs() { return 0; }

LogicalResult top::GridSamplerOp::init(InferenceParameter &p) { return success(); }

void top::GridSamplerOp::deinit(InferenceParameter &p) {}

LogicalResult top::GridSamplerOp::inference(InferenceParameter &p) {
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

void top::GridSamplerOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  auto grid_shape = module::getShape(getGrid());
  std::vector<int64_t> out_shape;
  out_shape.push_back(input_shape[0]);
  out_shape.push_back(input_shape[1]);
  out_shape.push_back(grid_shape[1]);
  out_shape.push_back(grid_shape[2]);
  auto out = getOutput();
  module::setShapeOrVerify(out, out_shape);

  // unsqueeze grid shape
  if (grid_shape.size() != input_shape.size() &&
      grid_shape[grid_shape.size() - 1] == 1) {
    std::vector<int64_t> new_shape(grid_shape.begin(),
                                   grid_shape.begin() + input_shape.size());
    auto newType = RankedTensorType::get(new_shape, module::getElementType(getGrid()));
    getGrid().setType(newType);
    // module::setShapeOrVerify(getGrid(), new_shape);
  }
  
}

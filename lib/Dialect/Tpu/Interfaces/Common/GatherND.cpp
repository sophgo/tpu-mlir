//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"

LogicalResult tpu::GatherNDOp::init(InferenceParameter &p) { return success(); }
void tpu::GatherNDOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::GatherNDOp::inference(InferenceParameter &p) {
  GatherNDParam param;
  param.batch_dims = getBatchDims();

  tensor_list_t input;
  input.ptr = p.inputs[0];
  input.shape = module::getShape(getInputData());
  input.size = module::getNumElements(getInputData());
  param.inputs.emplace_back(input);

  tensor_list_t indices;
  indices.ptr = p.inputs[1];
  indices.shape = module::getShape(getIndices());
  indices.size = module::getNumElements(getIndices());
  param.inputs.emplace_back(indices);

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

mlir::Type tpu::GatherNDOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  auto op = getOperation();
  if (opd_idx == 1) {
    // indices
    auto opd = op->getOperand(1);
    auto in_op = opd.getDefiningOp();
    if (in_op != nullptr && isa<top::WeightOp, top::NoneOp>(in_op)) {
      return do_nothing(mode);
    }
    auto stype = module::getStorageType(opd);
    if (stype.isIntOrIndex()) {
      return do_nothing(mode);
    }
    mode = TypeCastMode::DO_CAST;
    return Builder(op).getIntegerType(32);
  }
  return type_verify_case_same(op, opd_idx, mode);
}

bool tpu::GatherNDOp::support_multi_core() { return false; }

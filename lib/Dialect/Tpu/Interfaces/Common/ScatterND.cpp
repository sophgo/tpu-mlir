//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"

LogicalResult tpu::ScatterNDOp::init(InferenceParameter &p) {
  return success();
}
void tpu::ScatterNDOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ScatterNDOp::inference(InferenceParameter &p) {
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

mlir::Type tpu::ScatterNDOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
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

bool tpu::ScatterNDOp::support_multi_core() { return false; }

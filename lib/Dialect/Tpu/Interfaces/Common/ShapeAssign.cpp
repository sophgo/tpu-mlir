//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::ShapeAssignOp::init(InferenceParameter &p) {
  return success();
}
void tpu::ShapeAssignOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ShapeAssignOp::inference(InferenceParameter &p) {
  const int num_elem = module::getNumElements(getInput());
  std::copy(p.inputs[0], p.inputs[0] + num_elem, p.outputs[0]);
  return success();
}

mlir::Type tpu::ShapeAssignOp::type_verify(uint64_t opd_idx,
                                           TypeCastMode &mode) {
  auto op = getOperation();
  if (opd_idx == 1) {
    // shape
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

bool tpu::ShapeAssignOp::support_multi_core() { return false; }

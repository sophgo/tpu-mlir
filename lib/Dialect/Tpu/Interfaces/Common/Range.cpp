//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::RangeOp::init(InferenceParameter &p) { return success(); }
void tpu::RangeOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RangeOp::inference(InferenceParameter &p) {
  auto start = p.inputs[0][0];
  auto limit = p.inputs[1][0];
  auto delta = p.inputs[2][0];
  auto output = p.outputs[0];
  if(limit > start) {
    for (int i = 0, n = start; n < limit; n += delta, ++i)
      output[i] = n;
  } else {
    for (int i = 0, n = start; n > limit; n += delta, ++i)
      output[i] = n;
  }
  return success();
}

mlir::Type tpu::RangeOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  auto op = getOperation();
  auto opd = op->getOperand(opd_idx);
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

bool tpu::RangeOp::support_multi_core() { return false; }

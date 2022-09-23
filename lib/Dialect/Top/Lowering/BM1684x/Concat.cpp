//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

Value top::ConcatOp::lowering_int8_bm1684x(bool asymmetric) {
  auto op = getOperation();
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  std::vector<Value> operands;
  for (auto in : inputs()) {
    auto new_in = do_transfer(in, output(), asymmetric);
    operands.push_back(new_in);
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newType = Quant::getQuantInt8Type(output(), asymmetric);
  auto newOp =
      builder.create<tpu::ConcatOp>(getLoc(), newType, operands, attrs);
  return newOp.output();
}

Value top::ConcatOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::ConcatOp>(getOperation());
}

Value top::ConcatOp::lowering_bf16_bm1684x() {
  return lowering_common_float<tpu::ConcatOp, BFloat16Type>(getOperation());
}

Value top::ConcatOp::lowering_f16_bm1684x() {
  return lowering_common_float<tpu::ConcatOp, Float16Type>(getOperation());
}

Value top::ConcatOp::lowering_quant_bm1684x() {
  // if (!output().getType().isUnsignedInteger(8)) {
  //   return lowering_common<tpu::ConcatOp>(getOperation(), output().getType());
  // }
  auto op = getOperation();
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  std::vector<Value> operands;
  auto out_stype = Module::getStorageType(output());
  if (out_stype.isUnsignedInteger(8)) {
    for (auto in : inputs()) {
      auto new_in = do_transfer_fp(in, output(), true);
      operands.push_back(new_in);
    }
  } else {
    for (auto in : inputs()) {
      operands.push_back(in);
    }
  }

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newOp =
      builder.create<tpu::ConcatOp>(getLoc(), output().getType(), operands, attrs);
  return newOp.output();
}

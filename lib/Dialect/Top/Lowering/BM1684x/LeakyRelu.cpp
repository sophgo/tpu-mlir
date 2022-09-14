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

Value top::LeakyReluOp::lowering_int8_bm1684x(bool asymmetric) {
  if (!asymmetric) {
    auto op = getOperation();
    OpBuilder builder(op);
    Value input = op->getOperand(0);

    int multiplier, rshift;
    get_scale_and_shift(alpha().convertToDouble(), multiplier, rshift, 8);

    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr(
        "multiplier", builder.getI64IntegerAttr(multiplier)));
    attrs.push_back(
        builder.getNamedAttr("rshift", builder.getI64IntegerAttr(rshift)));

    builder.setInsertionPointAfter(op);
    auto newType = Quant::getQuantInt8Type(output(), asymmetric);
    auto newOp = builder.create<tpu::LeakyReluOp>(op->getLoc(), newType,
                                                  Value(input), attrs);
    return newOp;
  } else {
    return lowering_f32_bm1684x();
  }
  return nullptr;
}

Value top::LeakyReluOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::LeakyReluOp, Float32Type>(getOperation());
  return nullptr;
}

Value top::LeakyReluOp::lowering_bf16_bm1684x() {
  return lowering_common_float<tpu::LeakyReluOp, BFloat16Type>(getOperation());
}

Value top::LeakyReluOp::lowering_f16_bm1684x() {
  return lowering_common_float<tpu::LeakyReluOp, Float16Type>(getOperation());
}

Value top::LeakyReluOp::lowering_quant_bm1684x() {
  llvm_unreachable("Not supported now");
  return nullptr;
}

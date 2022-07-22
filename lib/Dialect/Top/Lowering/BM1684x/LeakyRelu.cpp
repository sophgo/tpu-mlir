//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

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
    attrs.push_back(builder.getNamedAttr("name", nameAttr()));
    attrs.push_back(builder.getNamedAttr(
        "multiplier", builder.getI64IntegerAttr(multiplier)));
    attrs.push_back(
        builder.getNamedAttr("rshift", builder.getI64IntegerAttr(rshift)));

    builder.setInsertionPointAfter(op);
    auto newType = Quant::getQuantInt8Type(output(), asymmetric);
    auto newOp = builder.create<tpu::LeakyReluOp>(
        op->getLoc(), newType, Value(input), ArrayRef<NamedAttribute>{attrs});
    return newOp;
  } else {
    llvm_unreachable("to be supported for LeakyRelu asymmetric quantize lowering");
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

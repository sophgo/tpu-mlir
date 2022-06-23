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
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

Value top::AddOp::lowering_int8_bm1684x(bool asymetric) {
  if (asymetric == false) {
    auto op = getOperation();
    OpBuilder builder(op);
    std::vector<Value> operands;
    const int nInputs = op->getNumOperands();
    std::vector<int64_t> rshift_v(nInputs);
    std::vector<int64_t> multiplier_v(nInputs, 1);
    int64_t o_zp;
    double o_scale;
    Quant::getScaleAndZeroPoint(output(), o_scale, o_zp, asymetric);
    auto coeff_v = Module::getF64Array(coeff(), nInputs, 1.0);

    double scale;
    int64_t zeropoint;
    for (int i = 0; i < nInputs; i++) {
      auto input = op->getOperand(i);
      operands.push_back(input);
      Quant::getScaleAndZeroPoint(input, scale, zeropoint, asymetric);
      int scalei, shifti;
      auto scale_f = scale / o_scale;
      get_scale_and_shift(coeff_v->at(i) * scale_f, scalei, shifti, 8);
      multiplier_v[i] = scalei;
      rshift_v[i] = shifti;
    }

    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("name", nameAttr()));
    attrs.push_back(builder.getNamedAttr("do_relu", do_reluAttr()));
    attrs.push_back(builder.getNamedAttr(
        "multipliers", builder.getI64ArrayAttr(multiplier_v)));
    attrs.push_back(
        builder.getNamedAttr("rshifts", builder.getI64ArrayAttr(rshift_v)));
    auto newType = Quant::getQuantInt8Type(output(), asymetric);
    auto newOp = builder.create<tpu::AddOp>(op->getLoc(), newType,
                                            ArrayRef<Value>{operands},
                                            ArrayRef<NamedAttribute>{attrs});
    return newOp.output();
  } else {
    llvm_unreachable("AddOp asymetric use f32");
  }
}

Value top::AddOp::lowering_f32_bm1684x() {
  return lowering_common<tpu::AddOp>(getOperation());
}

Value top::AddOp::lowering_bf16_bm1684x() {
  return lowering_common<tpu::AddOp, BFloat16Type>(getOperation());
}

Value top::AddOp::lowering_f16_bm1684x() {
  return lowering_common<tpu::AddOp, Float16Type>(getOperation());
}

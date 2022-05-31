//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Quant.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;

Value top::AddOp::lowering_int8_bm1686() {
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  std::vector<int64_t> rshift_v(nInputs);
  std::vector<int64_t> multiplier_v(nInputs, 1);
  int64_t o_zp;
  double o_scale;
  Quant::getScaleAndZeroPoint(output(), o_scale, o_zp);
  auto coeff_v = Module::getF64Array(coeff(), nInputs, 1.0);

  double bias = 0;
  int max_shifti = -32;
  double scale;
  int64_t zeropoint;
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    operands.push_back(input);
    Quant::getScaleAndZeroPoint(input, scale, zeropoint);
    int scalei, shifti;
    auto scale_f = scale / o_scale;
    bias += scale_f * zeropoint;
    get_scale_and_shift(coeff_v->at(i) * scale_f, scalei, shifti, 8);
    multiplier_v[i] = scalei;
    rshift_v[i] = shifti;
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu", do_reluAttr()));
  attrs.push_back(builder.getNamedAttr("multipliers",
                                       builder.getI64ArrayAttr(multiplier_v)));
  attrs.push_back(
      builder.getNamedAttr("rshifts", builder.getI64ArrayAttr(rshift_v)));
  attrs.push_back(
      builder.getNamedAttr("rectified_bias", builder.getF64FloatAttr(bias)));
  auto newOp = builder.create<tpu::AddOp>(op->getLoc(), output().getType(),
                                          ArrayRef<Value>{operands},
                                          ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(newOp.output(), true);
  return newOp.output();
}

Value top::AddOp::lowering_fp(llvm::StringRef mode) {
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  auto coeff_v = Module::getF64Array(coeff(), nInputs, 1.0);

  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    operands.push_back(input);
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu", do_reluAttr()));
  attrs.push_back(
      builder.getNamedAttr("coeff", builder.getF64ArrayAttr(*coeff_v)));

  auto newOp = builder.create<tpu::AddOp>(op->getLoc(), output().getType(),
                                          ArrayRef<Value>{operands},
                                          ArrayRef<NamedAttribute>{attrs});
  return newOp.output();
}

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
#include "mlir/IR/Location.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

Value top::AddOp::lowering_int8_bm1684x(bool asymmetric) {
  if (asymmetric == false) {
    auto op = getOperation();
    OpBuilder builder(op);
    std::vector<Value> operands;
    const int nInputs = op->getNumOperands();
    std::vector<int64_t> rshift_v(nInputs);
    std::vector<int64_t> multiplier_v(nInputs, 1);
    int64_t o_zp;
    double o_scale;
    Quant::getScaleAndZeroPoint(output(), o_scale, o_zp, asymmetric);
    auto coeff_v = Module::getF64Array(coeff(), nInputs, 1.0);

    double scale;
    int64_t zeropoint;
    for (int i = 0; i < nInputs; i++) {
      auto input = op->getOperand(i);
      operands.push_back(input);
      Quant::getScaleAndZeroPoint(input, scale, zeropoint, asymmetric);
      int scalei, shifti;
      auto scale_f = scale / o_scale;
      // get_scale_and_shift(coeff_v->at(i) * scale_f, scalei, shifti, 8);
      // "get_scale_and_shift_positive" use positive right_shift, left_shift
      // will be converted to the multiplier.
      get_scale_and_shift_positive(coeff_v->at(i) * scale_f, scalei, shifti, 8);
      multiplier_v[i] = scalei;
      rshift_v[i] = shifti;
    }

    builder.setInsertionPointAfter(op);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("do_relu", do_reluAttr()));
    attrs.push_back(builder.getNamedAttr(
        "multipliers", builder.getI64ArrayAttr(multiplier_v)));
    attrs.push_back(
        builder.getNamedAttr("rshifts", builder.getI64ArrayAttr(rshift_v)));
    auto newType = Quant::getQuantInt8Type(output(), asymmetric);
    auto newOp = builder.create<tpu::AddOp>(op->getLoc(), newType,
                                            ArrayRef<Value>{operands},
                                            ArrayRef<NamedAttribute>{attrs});
    return newOp.output();
  } else {
    llvm_unreachable("AddOp asymmetric use f32");
  }
}

Value top::AddOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::AddOp>(getOperation());
}

Value top::AddOp::lowering_bf16_bm1684x() {
  return lowering_common_float<tpu::AddOp, BFloat16Type>(getOperation());
}

Value top::AddOp::lowering_f16_bm1684x() {
  return lowering_common_float<tpu::AddOp, Float16Type>(getOperation());
}

Value top::AddOp::lowering_quant_bm1684x() {
  if (Quant::isUniformQuantized(inputs()[0], output()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  auto op = getOperation();
  OpBuilder builder(getContext());
  const int nInputs = op->getNumOperands();
  assert(nInputs==2);     // TODO: nInput==1
  const int nTensors = nInputs + 1;
  const int lshift = 20;  // TODO: lshift == 15 if input dtype is int16
  std::vector<int64_t> shift_v(nTensors);
  std::vector<int64_t> multiplier_v(nTensors, 1);
  std::vector<double> scale_v(nInputs);
  int64_t zeropoint;
  double o_scale;
  Quant::getScaleAndZeroPoint(output(), o_scale, zeropoint, true);

  // generate quant param from given scale
  double scale, scale_max;
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    Quant::getScaleAndZeroPoint(input, scale, zeropoint, true);
    scale_v[i] = scale;
    if (i == 0) {
      scale_max = scale;
    } else {
      scale_max = scale > scale_max ? scale : scale_max;
    }
  }
  int64_t scalei, shifti;
  for (int i = 0; i < nInputs; i++) {
    auto scale_f = scale_v[i] / (scale_max * 2);
    QuantizeMultiplier(scale_f, &scalei, &shifti);
    multiplier_v[i] = scalei;
    shift_v[i] = shifti;
  }

  std::vector<Value> operands;
  auto ctx = op->getContext();

  // dequant left
  auto input0_dequant = do_dequant(inputs()[0], builder.getI32Type(), multiplier_v[0], shift_v[0], 1, lshift);
  // op->setOperand(0, input0_dequant);
  operands.push_back(input0_dequant);
  // dequant right
  auto input1_dequant = do_dequant(inputs()[1], builder.getI32Type(), multiplier_v[1], shift_v[1], 1, lshift);
  // op->setOperand(1, input1_dequant);
  operands.push_back(input1_dequant);
  // add
  std::string suffix = "_add";
  std::string new_name = Module::getName(op).str() + suffix;
  builder.setInsertionPointAfterValue(input1_dequant);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr(
      "multiplier", builder.getI64IntegerAttr(1)));
  attrs.push_back(
      builder.getNamedAttr("shift", builder.getI64IntegerAttr(0)));

  auto newType = RankedTensorType::get(Module::getShape(output()), builder.getI32Type());
  // auto add_quant = lowering_common<tpu::AddOp>(op, newType);
  auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
  auto newOp =
      builder.create<tpu::AddOp>(name_loc, newType, ArrayRef<Value>{operands},
                                 ArrayRef<NamedAttribute>{attrs});
  // requant to int8
  QuantizeMultiplier((scale_max * 2) / ((1 << lshift) * o_scale), &scalei, &shifti);
  return do_requant(op->getLoc(), newOp.output(), output().getType(), true, scalei, shifti, 1);
}

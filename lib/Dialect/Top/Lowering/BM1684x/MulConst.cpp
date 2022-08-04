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
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

static Value asym_transfer(Value in, Value out, double const_val) {
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  Quant::getScaleAndZeroPoint(in, in_scale, in_zp, 1);
  Quant::getScaleAndZeroPoint(out, out_scale, out_zp, 1);
  if (in_scale == out_scale && in_zp == out_zp) {
    return in;
  }
  auto in_shape = Module::getShape(in);
  auto out_type = Quant::getQuantInt8Type(out, 1);
  auto ele_type = out_type.cast<RankedTensorType>().getElementType();
  auto new_type = RankedTensorType::get(in_shape, ele_type);

  auto op = out.getDefiningOp();
  OpBuilder builder(op);
  auto in_name = Module::getName(in.getDefiningOp());
  auto out_name = Module::getName(op);
  auto new_name = in_name + "_to_" + out_name;
  int multiplier, rshift;
  get_scale_and_shift(in_scale / out_scale * const_val, multiplier, rshift, 8);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(
        builder.getNamedAttr("name", builder.getStringAttr(new_name)));
  attrs.push_back(builder.getNamedAttr(
        "multiplier", builder.getI64IntegerAttr(multiplier)));
  attrs.push_back(
        builder.getNamedAttr("rshift", builder.getI64IntegerAttr(rshift)));
  attrs.push_back(
        builder.getNamedAttr("quant_mode", builder.getI64IntegerAttr(2)));
  auto in_type = in.getType().cast<RankedTensorType>();
  auto none = Module::getNoneOp(op);
  builder.setInsertionPointAfterValue(in);
  auto rqOp = builder.create<tpu::RequantOp>(op->getLoc(), new_type,
                                             ValueRange{in, none},
                                             ArrayRef<NamedAttribute>{attrs});
  return rqOp.output();
}

Value top::MulConstOp::lowering_int8_bm1684x(bool asymmetric) {
  auto op = getOperation();
  const int nInputs = op->getNumOperands();
  if (!asymmetric) {
    OpBuilder builder(op);
    double scale_i, scale_o;
    int64_t zp_i, zp_o;
    double thBottom, thTop;

    Quant::getScaleAndZeroPoint(input(), scale_i, zp_i, asymmetric);
    Quant::getScaleAndZeroPoint(output(), scale_o, zp_o, asymmetric);

    auto scale = scale_i / scale_o * const_val().convertToDouble();

    int multiplier, rshift;
    get_scale_and_shift(scale, multiplier, rshift, 8);

    std::vector<NamedAttribute> attrs;
    for (auto &attr : op->getAttrs()) {
      attrs.push_back(attr);
    }
    attrs.push_back(builder.getNamedAttr(
        "multiplier", builder.getI64IntegerAttr(multiplier)));
    attrs.push_back(
        builder.getNamedAttr("rshift", builder.getI64IntegerAttr(rshift)));
    auto newType = Quant::getQuantInt8Type(output(), asymmetric);
    auto newOp = builder.create<tpu::MulConstOp>(
        op->getLoc(), newType, ValueRange{input()},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.output();
  } else {
    OpBuilder builder(op);
    auto new_input = asym_transfer(input(), output(), const_val().convertToDouble());
    std::vector<NamedAttribute> attrs;
    for (auto &attr : op->getAttrs()) {
      attrs.push_back(attr);
    }
    auto newType = Quant::getQuantInt8Type(output(), asymmetric);
    auto newOp = builder.create<tpu::MulConstOp>(
        op->getLoc(), newType, ValueRange{new_input},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.output();
  }
}

Value top::MulConstOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::MulConstOp, Float32Type>(getOperation());
}

Value top::MulConstOp::lowering_bf16_bm1684x() {
  return lowering_common_float<tpu::MulConstOp, BFloat16Type>(getOperation());
}

Value top::MulConstOp::lowering_f16_bm1684x() {
  return lowering_common_float<tpu::MulConstOp, Float16Type>(getOperation());
}

Value top::MulConstOp::lowering_quant_bm1684x() {
  Builder builder(getContext());
  auto in0_f32 = do_cast(input(), builder.getF32Type(), false);
  auto op = getOperation();
  op->setOperand(0, in0_f32);
  auto type = output().getType();
  auto v = lowering_common_float<tpu::MulConstOp>(op);
  return do_cast(v, type, true);
}

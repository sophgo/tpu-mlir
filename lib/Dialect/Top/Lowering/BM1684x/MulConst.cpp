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

Value top::MulConstOp::lowering_int8_bm1684x(bool asymmetric) {
  auto op = getOperation();
  const int nInputs = op->getNumOperands();
  if (!asymmetric) {
    OpBuilder builder(op);
    std::vector<Value> operands;
    operands.push_back(op->getOperand(0));
    double scale_i, scale_o;
    int64_t zp_i, zp_o;
    double thBottom, thTop;

    Quant::getScaleAndZeroPoint(op->getOperand(0), scale_i, zp_i, asymmetric);
    Quant::getScaleAndZeroPoint(output(), scale_o, zp_o, asymmetric);

    auto scale = scale_i / scale_o * coeff().convertToDouble();

    int multiplier, rshift;
    get_scale_and_shift(scale, multiplier, rshift, 8);

    //thBottom = scale_i * 127.0;
    //thTop = scale_i * 127.0;

    //int rightShiftTmp = calRightShiftNumUseCblas(coeff().convertToDouble(), thBottom, thTop, 8);
    //double scale = std::pow(2.0, float(rightShiftTmp)) * thBottom / thTop;

    //int coeff_int8;
    //quantizeToInt8(&coeff, &coeff_int8, 1, scale);

    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("name", nameAttr()));
    attrs.push_back(builder.getNamedAttr("do_relu", do_reluAttr()));
    attrs.push_back(builder.getNamedAttr("multiplier", builder.getI64IntegerAttr(multiplier)));
    attrs.push_back(builder.getNamedAttr("rshift", builder.getI64IntegerAttr(rshift)));
    auto newType = Quant::getQuantInt8Type(output(), asymmetric);
    auto newOp = builder.create<tpu::MulConstOp>(op->getLoc(), newType,
                                                 ArrayRef<Value>{operands},
                                                 ArrayRef<NamedAttribute>{attrs});
    return newOp.output();
  } else {
    llvm_unreachable("MulConstOp asymmetric use FP32");
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
  auto in0_f32 = do_cast(inputs()[0], builder.getF32Type(), false);
  auto op = getOperation();
  op->setOperand(0, in0_f32);
  auto type = output().getType();
  auto v = lowering_common_float<tpu::MulConstOp>(op);
  return do_cast(v, type, true);
}

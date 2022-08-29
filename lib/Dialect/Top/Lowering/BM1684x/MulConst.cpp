//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
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
  OpBuilder builder(op);
  double scale_i, scale_o;
  int64_t zp_i, zp_o;
  Quant::getScaleAndZeroPoint(input(), scale_i, zp_i, asymmetric);
  Quant::getScaleAndZeroPoint(output(), scale_o, zp_o, asymmetric);
  auto scale = scale_i / scale_o * const_val().convertToDouble();
  int multiplier, rshift;
  get_scale_and_shift(scale, multiplier, rshift, 8);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(builder.getNamedAttr("multiplier",
                                       builder.getI64IntegerAttr(multiplier)));
  attrs.push_back(
      builder.getNamedAttr("rshift", builder.getI64IntegerAttr(rshift)));
  auto newType = Quant::getQuantInt8Type(output(), asymmetric);
  builder.setInsertionPointAfter(op);
  auto newOp = builder.create<tpu::MulShiftOp>(op->getLoc(), newType,
                                               ValueRange{input()}, attrs);
  return newOp.output();
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

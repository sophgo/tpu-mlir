//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

Value top::PadOp::lowering_int8_bm1684x(bool asymmetric) {
  auto op = getOperation();
  OpBuilder builder(op);
  int64_t in_zp;
  double in_scale;
  Quant::getScaleAndZeroPoint(input(), in_scale, in_zp, asymmetric);

  std::vector<NamedAttribute> attrs;
  auto val_ = val().convertToDouble();
  val_ = std::round(val_ / in_scale + in_zp);
  attrs.push_back(builder.getNamedAttr("paddings", paddingsAttr()));
  attrs.push_back(builder.getNamedAttr("val", builder.getF64FloatAttr(val_)));

  builder.setInsertionPointAfter(op);
  auto newType = Quant::getQuantInt8Type(output(), asymmetric);
  auto newOp = builder.create<tpu::PadOp>(op->getLoc(), newType,
                                          op->getOperands(), attrs);
  return newOp.output();
}

Value top::PadOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::PadOp>(getOperation());
}

Value top::PadOp::lowering_bf16_bm1684x() {
  return lowering_common_float<tpu::PadOp, BFloat16Type>(getOperation());
}

Value top::PadOp::lowering_f16_bm1684x() {
  return lowering_common_float<tpu::PadOp, Float16Type>(getOperation());
}

Value top::PadOp::lowering_quant_bm1684x() {
  llvm_unreachable("not support now");
}

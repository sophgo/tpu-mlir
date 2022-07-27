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

Value top::ConcatOp::lowering_int8_bm1684x(bool asymmetric) {
  auto op = getOperation();
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  double out_scale;
  int64_t out_zp;
  Quant::getScaleAndZeroPoint(output(), out_scale, out_zp, asymmetric);
  std::vector<Value> operands;
  auto newType = Quant::getQuantInt8Type(output(), asymmetric);
  auto eleType = newType.cast<RankedTensorType>().getElementType();
  auto concat_name = name();
  for (auto in : inputs()) {
    double in_scale;
    int64_t in_zp;
    Quant::getScaleAndZeroPoint(in, in_scale, in_zp, asymmetric);
    if (asymmetric == true || (in_scale == out_scale && in_zp == out_zp)) {
      operands.push_back(in);
    } else {
      int multiplier, rshift;
      get_scale_and_shift(in_scale / out_scale, multiplier, rshift, 8);
      std::vector<NamedAttribute> mr_attrs;
      auto name = Module::getName(in.getDefiningOp());
      mr_attrs.push_back(builder.getNamedAttr(
          "name", builder.getStringAttr(name + "_rq_" + concat_name)));
      mr_attrs.push_back(builder.getNamedAttr(
          "multiplier", builder.getI64IntegerAttr(multiplier)));
      mr_attrs.push_back(
          builder.getNamedAttr("rshift", builder.getI64IntegerAttr(rshift)));
      auto in_type = in.getType().cast<RankedTensorType>();
      auto in_shape = in_type.getShape();
      auto new_type = RankedTensorType::get(in_shape, eleType);
      auto mrOp =
          builder.create<tpu::MulShiftOp>(getLoc(), new_type, ValueRange{in},
                                          ArrayRef<NamedAttribute>{mr_attrs});
      operands.push_back(mrOp.output());
    }
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newOp = builder.create<tpu::ConcatOp>(getLoc(), newType,
                                             ArrayRef<Value>{operands},
                                             ArrayRef<NamedAttribute>{attrs});
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
  return lowering_common<tpu::ConcatOp>(getOperation(), output().getType());
}

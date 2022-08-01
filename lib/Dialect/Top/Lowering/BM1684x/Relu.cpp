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

Value top::ReluOp::lowering_int8_bm1684x(bool asymmetric) {
  auto operation = getOperation();
  auto newType = Quant::getQuantInt8Type(operation->getResult(0), asymmetric);
  auto ctx = operation->getContext();
  OpBuilder builder(ctx);
  std::vector<Value> operands;
  const int nInputs = operation->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(operation->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : operation->getAttrs()) {
    if (attr.getName() == "upper_limit") {
      float relu_upper_limit = upper_limitAttr().getValueAsDouble();
      if (relu_upper_limit <= 0.f)
        continue;
      double scale;
      int64_t out_zp;
      Quant::getScaleAndZeroPoint(output(), scale, out_zp, asymmetric);
      int upper_limit_quant = int(relu_upper_limit / scale + out_zp);
      attrs.push_back(builder.getNamedAttr("upper_limit", builder.getF64FloatAttr(upper_limit_quant)));
    } else {
      attrs.push_back(attr);
    }
  }

  builder.setInsertionPointAfter(operation);
  auto newOp =
      builder.create<tpu::ReluOp>(operation->getLoc(), newType, ArrayRef<Value>{operands},
                                  ArrayRef<NamedAttribute>{attrs});
  return newOp.output();
}

Value top::ReluOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::ReluOp>(getOperation());
}

Value top::ReluOp::lowering_bf16_bm1684x() {
  return lowering_common_float<tpu::ReluOp, BFloat16Type>(getOperation());
}

Value top::ReluOp::lowering_f16_bm1684x() {
  return lowering_common_float<tpu::ReluOp, Float16Type>(getOperation());
}

Value top::ReluOp::lowering_quant_bm1684x() {
  llvm_unreachable("not support now");
}

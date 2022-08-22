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

Value top::SoftmaxOp::lowering_int8_bm1684x(bool asymmetric) {
  return lowering_common_float<tpu::SoftmaxOp>(
      getOperation()); // skip int8 quant for now
}

Value top::SoftmaxOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::SoftmaxOp>(getOperation());
}

Value top::SoftmaxOp::lowering_bf16_bm1684x() {
  llvm_unreachable("to be supported for Softmax bf16 quantize lowering");
}

Value top::SoftmaxOp::lowering_f16_bm1684x() {
  llvm_unreachable("to be supported for Softmax f16 quantize lowering");
}

Value top::SoftmaxOp::lowering_quant_bm1684x() {
  if (Quant::isUniformQuantized(input(), output()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  auto op = getOperation();
  OpBuilder builder(getContext());
  const int nInputs = op->getNumOperands();
  int64_t zeropoint;
  double i_scale;
  Quant::getScaleAndZeroPoint(input(), i_scale, zeropoint, true);
  SmallVector<double> table(256);
  double beta_v = betaAttr().getValueAsDouble() ;
  double scale = -i_scale * beta_v;
  for (int i = 0; i < 256; ++i) {
    table[i] = std::exp(scale * i);
  }

  builder.setInsertionPointAfter(op);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("axis", axisAttr()));
  attrs.push_back(builder.getNamedAttr(
      "table", builder.getF64ArrayAttr(table)));
  auto newOp = builder.create<tpu::SoftmaxOp>(output().getLoc(), output().getType(),
                                              ValueRange{op->getOperand(0)},
                                              ArrayRef<NamedAttribute>{attrs});
  return newOp.output();
}

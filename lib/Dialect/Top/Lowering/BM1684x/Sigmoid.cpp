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

double active_sigmoid(double val) { return 1 / (1 + std::exp(-val)); }

Value top::SigmoidOp::lowering_int8_bm1684x(bool asymmetric) {
  auto op = getOperation();
  OpBuilder builder(op);
  auto stype = Module::getStorageType(output());
  Value table =
      create_lookup_table(input(), output(), active_sigmoid, asymmetric);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  builder.setInsertionPointAfter(op);
  auto newType = Quant::getQuantInt8Type(output(), asymmetric);
  auto newOp = builder.create<tpu::LutOp>(getLoc(), newType,
                                          ValueRange{input(), table}, attrs);
  return newOp.output();
}

Value top::SigmoidOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::SigmoidOp>(getOperation());
}

Value top::SigmoidOp::lowering_bf16_bm1684x() {
  return lowering_common_float<tpu::SigmoidOp, Float32Type>(getOperation());
}

Value top::SigmoidOp::lowering_f16_bm1684x() {
  return lowering_common_float<tpu::SigmoidOp, Float32Type>(getOperation());
}

Value top::SigmoidOp::lowering_quant_bm1684x() {
  llvm_unreachable("SigmoidOp not support now");
}

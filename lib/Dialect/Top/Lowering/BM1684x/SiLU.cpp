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

double active_silu(double val) { return val / (1 + std::exp(-val)); }

Value top::SiLUOp::lowering_int8_bm1684x(bool asymmetric) {
  auto ctx = getContext();
  auto op = getOperation();
  OpBuilder builder(ctx);
  auto stype = Module::getStorageType(output());
  auto table = create_lookup_table(input(), output(), active_silu, asymmetric);
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

Value top::SiLUOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::SiLUOp>(getOperation());
}

Value top::SiLUOp::lowering_bf16_bm1684x() {
  return lowering_common_float<tpu::SiLUOp, Float32Type>(getOperation());
}

Value top::SiLUOp::lowering_f16_bm1684x() {
  return lowering_common_float<tpu::SiLUOp, Float32Type>(getOperation());
}

Value top::SiLUOp::lowering_quant_bm1684x() {
  llvm_unreachable("SiLUOp not support now");
}

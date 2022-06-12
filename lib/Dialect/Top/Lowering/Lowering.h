//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir::helper;
namespace tpu_mlir {
namespace top {

template <typename OpTy, typename ElemTy = Float32Type>
static mlir::Value lowering_common(Operation *from) {
  auto ctx = from->getContext();
  OpBuilder builder(ctx);
  std::vector<Value> operands;
  const int nInputs = from->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(from->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : from->getAttrs()) {
    attrs.push_back(attr);
  }
  builder.setInsertionPointAfter(from);
  auto resultType = from->getResult(0).getType().cast<RankedTensorType>();
  auto newType = RankedTensorType::get(resultType.getShape(), ElemTy::get(ctx));
  auto newOp =
      builder.create<OpTy>(from->getLoc(), newType, ArrayRef<Value>{operands},
                           ArrayRef<NamedAttribute>{attrs});
  return newOp.output();
}

} // namespace top
} // namespace tpu_mlir

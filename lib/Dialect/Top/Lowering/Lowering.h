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
  auto output = from->getResult(0);
  auto sType = Module::getStorageType(output);
  auto shape = Module::getShape(output);
  Type newType = output.getType();
  if (sType.isa<ElemTy>() == false) {
    if (Quant::isCalibratedType(output)) {
      auto caliType = Quant::getCalibratedType(output);
      auto newCaliType = quant::CalibratedQuantizedType::get(
          ElemTy::get(ctx), caliType.getMin(), caliType.getMax());
      newType = RankedTensorType::get(shape, newCaliType);
    } else {
      newType = RankedTensorType::get(shape, ElemTy::get(ctx));
    }
  }
  auto newOp =
      builder.create<OpTy>(from->getLoc(), newType, ArrayRef<Value>{operands},
                           ArrayRef<NamedAttribute>{attrs});
  return newOp.output();
}

template <typename OpTy>
static mlir::Value lowering_common_int8(Operation *from,
                                        bool asymetric = false) {
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
  auto newType = Quant::getQuantInt8Type(from->getResult(0), asymetric);
  auto newOp =
      builder.create<OpTy>(from->getLoc(), newType, ArrayRef<Value>{operands},
                           ArrayRef<NamedAttribute>{attrs});
  return newOp.output();
}

} // namespace top
} // namespace tpu_mlir

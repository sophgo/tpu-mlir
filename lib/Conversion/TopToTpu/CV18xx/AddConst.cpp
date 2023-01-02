//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-add-const"

namespace tpu_mlir {
namespace cv18xx {

void lowering_add_const(PatternRewriter &rewriter, top::AddConstOp op) {
  std::vector<Value> operands;
  std::vector<float> weight_data;

  weight_data.emplace_back(op.getConstVal().convertToDouble());
  auto weight_type = RankedTensorType::get({1}, rewriter.getF32Type());
  auto weight_operand =
      top::WeightOp::create(op, "const_val", weight_data, weight_type);
  operands.emplace_back(op.getInput());
  operands.emplace_back(weight_operand);

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  rewriter.replaceOpWithNewOp<top::AddOp>(
      op, op.getOutput().getType().cast<RankedTensorType>(), operands, attrs);
  return;
}

void AddConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::AddConstOp op, bool asymmetric) const {
  lowering_add_const(rewriter, op);
}

void AddConstLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::AddConstOp op) const {
  lowering_add_const(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir

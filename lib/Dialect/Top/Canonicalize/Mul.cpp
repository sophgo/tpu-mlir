//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace tpu_mlir::top;

struct MulToSiLU : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    if (op.do_relu() || op.inputs().size() != 2) {
      return failure();
    }
    auto in0_op = op.inputs()[0].getDefiningOp();
    auto in1_op = op.inputs()[1].getDefiningOp();
    Value in_value;
    SigmoidOp sigmoid_op = dyn_cast<SigmoidOp>(in1_op);
    if (sigmoid_op && sigmoid_op.input().getDefiningOp() == in0_op &&
        sigmoid_op->hasOneUse()) {
      in_value = op.inputs()[0];
    } else if ((sigmoid_op = dyn_cast<SigmoidOp>(in0_op)) &&
               sigmoid_op.input().getDefiningOp() == in1_op &&
               sigmoid_op->hasOneUse()) {
      in_value = op.inputs()[1];
    } else {
      return failure();
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", op.nameAttr()));
    rewriter.replaceOpWithNewOp<SiLUOp>(op, op.getResult().getType(),
                                        ValueRange{in_value}, attrs);
    return success();
  }
};

void MulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<MulToSiLU>(context);
}

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace tpu_mlir::top;

// MatMul + Add(weight) => MatMul
struct MatMulWithBias : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto filter = op.getRight();
    if (module::isWeight(filter) == false) {
      return failure();
    }
    if (module::isNone(op.getBias()) == false) {
      return failure();
    }
    if (op->hasOneUse() == false) {
      return failure();
    }
    auto user = *op->getUsers().begin();
    auto add_op = dyn_cast<AddOp>(user);
    if (!user) {
      return failure();
    }
    if (add_op.getNumOperands() != 2) {
      return failure();
    }
    Value bias = nullptr;
    bool bias_is_weight = false;
    for (auto v : add_op.getOperands()) {
      if (module::isWeight(v)) {
        bias = v;
        bias_is_weight = true;
        break;
      }
    }
    if (!bias_is_weight) {
      return failure();
    }
    auto p = op.parseParam();
    if (p.batch > 1) {
      // TODO: not support batch matmul; need to support
      return failure();
    }
    if (module::getNumElements(bias) != p.N) {
      // TODO: maybe == 1 is OK
      return failure();
    }
    auto bias_op = bias.getDefiningOp();
    bias_op->moveBefore(op);
    op->setOperand(2, bias);
    op->setLoc(add_op.getLoc());
    add_op.replaceAllUsesWith(op.getOperation());
    add_op.erase();
    return success();
  }
};

void MatMulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<MatMulWithBias>(context);
}

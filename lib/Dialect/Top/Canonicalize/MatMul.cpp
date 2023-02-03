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
#include "tpu_mlir/Support/MathUtils.h"
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
    if (!add_op) {
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

// transpose + matmul
struct MatMulWithRightTranspose : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto filter = op.getRight();
    if (module::isWeight(filter)) {
      return failure();
    }
    if (filter.hasOneUse() == false) {
      return failure();
    }
    auto trans_op = dyn_cast<PermuteOp>(filter.getDefiningOp());
    if (!trans_op) {
      return failure();
    }
    auto attr = op.parseParam();
    int to_dim = 2;
    if (attr.batch > 1) {
      to_dim = 3;
    }
    std::vector<int64_t> shape = module::getShape(trans_op.getInput());
    auto order = module::getI64Array(trans_op.getOrder());
    std::vector<int64_t> shape_fix;
    std::vector<int64_t> order_fix;
    auto ret = permute_reset(shape, *order, shape_fix, order_fix, to_dim);
    if (ret == false) {
      return failure();
    }
    int n_idx = to_dim - 2;
    int k_idx = to_dim - 1;
    if (shape_fix[n_idx] == attr.N && shape_fix[k_idx] == attr.K &&
        order_fix[n_idx] == k_idx && order_fix[k_idx] == n_idx) {
      // bingo !
      op.setOperand(1, trans_op.getInput());
      op.setRightTranspose(true);
      trans_op.erase();
      return success();
    }
    return failure();
  }
};

void MatMulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<MatMulWithBias, MatMulWithRightTranspose>(context);
}

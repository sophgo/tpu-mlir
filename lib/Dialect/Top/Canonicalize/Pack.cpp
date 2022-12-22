//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Helper/Module.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace tpu_mlir::top;
using namespace tpu_mlir::helper;

// for bert model, concat matmuls to batch matmul
struct PackMatmulPattern : public OpRewritePattern<PackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PackOp op,
                                PatternRewriter &rewriter) const override {
    Operation *l_split_op = nullptr;
    Operation *r_split_op = nullptr;
    if (!op->hasOneUse()) {
      return failure();
    }
    for (auto in : op.inputs()) {
      auto matmul = dyn_cast<MatMulOp>(in.getDefiningOp());
      if (!matmul || !matmul->hasOneUse()) {
        return failure();
      }
      auto the_split_op = matmul.input().getDefiningOp();
      auto split = dyn_cast<SplitOp>(the_split_op);
      if (!split) {
        return failure();
      } else if (l_split_op == nullptr) {
        l_split_op = the_split_op;
      } else if (l_split_op != the_split_op) {
        return failure();
      }
      auto permute = dyn_cast<PermuteOp>(matmul.right().getDefiningOp());
      if (!permute || !permute->hasOneUse()) {
        return failure();
      }
      auto reshape = dyn_cast<ReshapeOp>(permute.input().getDefiningOp());
      if (!reshape || !reshape->hasOneUse()) {
        return failure();
      }
      the_split_op = reshape.input().getDefiningOp();
      split = dyn_cast<SplitOp>(the_split_op);
      if (!split) {
        return failure();
      } else if (r_split_op == nullptr) {
        r_split_op = the_split_op;
      } else if (r_split_op != the_split_op) {
        return failure();
      }
    }
    if (!l_split_op || !r_split_op) {
      return failure();
    }
    auto l_split = cast<SplitOp>(l_split_op);
    auto r_split = cast<SplitOp>(r_split_op);
    rewriter.setInsertionPointAfter(op);
    auto none = Module::getNoneOp(op);
    std::vector<NamedAttribute> attrs;
    if (op.do_relu()) {
      attrs.push_back(rewriter.getNamedAttr("do_relu", op.do_reluAttr()));
      attrs.push_back(rewriter.getNamedAttr("relu_limit", op.relu_limitAttr()));
    }
    auto batchMatMul = rewriter.create<MatMulOp>(
        op.getLoc(), op.output().getType(),
        ValueRange{l_split.input(), r_split.input(), none}, attrs);
    op.replaceAllUsesWith(batchMatMul.getOperation());
    return success();
  }
};

void PackOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<PackMatmulPattern>(context);
}

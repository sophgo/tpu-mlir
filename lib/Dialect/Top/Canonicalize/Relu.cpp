//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct TopFuseRelu : public OpRewriterPatternEx<ReluOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  TopFuseRelu(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ReluOp>(context, "TopFuseRelu") {}

  LogicalResult matchAndRewriteImpl(ReluOp op,
                                    PatternRewriter &rewriter) const override {
    auto formerOp = op.getInput().getDefiningOp();
    if (!formerOp->getResult(0).hasOneUse())
      return failure();

    if (false == formerOp->hasTrait<SupportFuseRelu>()) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
      return failure();
    }
    auto relu_limit = op.getReluLimit().convertToDouble();
    if (formerOp->hasAttr("relu_limit")) {
      auto old_limit =
          formerOp->getAttr("relu_limit").cast<FloatAttr>().getValueAsDouble();
      if (old_limit > 0 && relu_limit > old_limit) {
        relu_limit = old_limit;
      }
    }
    formerOp->setAttr("do_relu", rewriter.getBoolAttr(true));
    formerOp->setAttr("relu_limit", rewriter.getF64FloatAttr(relu_limit));
    bool if_multi_output = formerOp->getNumResults() > 1;
    if (if_multi_output) {
      std::vector<Location> locs;
      locs.push_back(op.getLoc());
      for (int i = 1; i < formerOp->getNumResults(); i++) {
        auto value = formerOp->getResult(i);
        auto loc = tpu_mlir::module::getLoc(value);
        locs.push_back(loc);
      }
      auto new_loc = FusedLoc::get(module::getCtx(), locs);
      formerOp->setLoc(new_loc);
    } else {
      formerOp->setLoc(op.getLoc());
    }
    // remove the relu Op
    rewriter.replaceOp(op, {op.getInput()});
    return success();
  }
};

struct TopMoveReluAheadConcatPattern : public OpRewriterPatternEx<ReluOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopMoveReluAheadConcatPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ReluOp>(context, "TopMoveReluAheadConcatPattern") {}

  LogicalResult matchAndRewriteImpl(ReluOp op,
                                    PatternRewriter &rewriter) const override {
    std::string op_name = op.getOperationName().str();
    auto relu_limit = op.getReluLimit();
    // match relu Op that is following concat Ops
    auto formerOp = op->getOperand(0).getDefiningOp();
    if (!formerOp->hasOneUse() || !isa<ConcatOp>(formerOp)) {
      return failure();
    }

    auto concatOp = cast<ConcatOp>(formerOp);
    int num_inputs = concatOp.getInputs().size();
    rewriter.setInsertionPoint(formerOp);
    for (int i = 0; i < num_inputs; i++) {
      auto inOp = formerOp->getOperand(i).getDefiningOp();
      if (false == inOp->hasTrait<SupportFuseRelu>()) {
        return failure();
      }
      auto inOp_name = module::getName(inOp).str();
      std::string new_name = inOp_name + "_move_ahead_relu";
      auto nameAttr = rewriter.getStringAttr(new_name);
      auto newOp = rewriter.create<ReluOp>(
          NameLoc::get(nameAttr), formerOp->getOperand(i).getType(),
          ArrayRef<Value>{formerOp->getOperand(i)});
      formerOp->setOperand(i, newOp.getResult());
    }

    // change the concat Op's name to avoid comparison between concat before and
    // after relu
    concatOp->setLoc(NameLoc::get(
        rewriter.getStringAttr(module::getName(formerOp).str() + "_relu")));

    rewriter.replaceOp(op, concatOp);
    return success();
  }
};

void ReluOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<TopMoveReluAheadConcatPattern, TopFuseRelu>(context);
}

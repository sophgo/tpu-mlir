//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/Module.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <set>
#include <sstream>

using namespace llvm;


namespace tpu_mlir {
namespace tpu {

// make sure operands is nearest to owner op
struct OpReorderPattern : public RewritePattern {
  OpReorderPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (isa<FuncOp, top::WeightOp, top::NoneOp>(op)) {
      return failure();
    }
    llvm::SmallVector<Operation *, 8> opds;
    llvm::SmallVector<Operation *, 8> weights;
    for (auto opd : op->getOperands()) {
      auto op_ = opd.getDefiningOp();
      if (op_ == nullptr || isa<top::NoneOp, FuncOp>(op_)) {
        continue;
      }
      if (isa<top::WeightOp>(op_)) {
        weights.push_back(op_);
      } else {
        if (op_->hasOneUse() == true) {
          opds.push_back(op_);
        }
      }
    }
    opds.append(weights);
    bool fixed = false;
    auto last_op = op;
    auto num_opds = opds.size();
    if (num_opds != 0) {
      for (auto it = opds.rbegin(); it != opds.rend(); ++it) {
        if ((*it)->getNextNode() != last_op) {
          (*it)->moveBefore(last_op);
          fixed = true;
        }
        last_op = *it;
      }
    }
    return fixed ? success() : failure();
  }
};

// test by sd_decoder_pt
struct AttentionReorderPattern : public RewritePattern {
  AttentionReorderPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (isa<FuncOp, top::WeightOp, top::NoneOp>(op)) {
      return failure();
    }
    if (op->getNumResults() != 1) {
      return failure();
    }
    auto out = op->getResult(0);
    auto num_uses = std::distance(out.use_begin(), out.use_end());
    if (num_uses < 2) {
      return failure();
    }
    // collect ops
    llvm::SmallVector<Operation *, 16> mm_ops;
    llvm::SmallVector<Operation *, 16> reshape_ops;
    for (auto user : out.getUsers()) {
      if (false == isa<tpu::MatMulOp>(user)) {
        return failure();
      }
      if (user->hasOneUse() == false) {
        return failure();
      }
      auto user2 = *user->getUsers().begin();
      if (false == isa<tpu::ReshapeOp>(user2)) {
        return failure();
      }
      for (auto opd : user->getOperands()) {
        if (opd == out) {
          continue;
        }
        auto op_ = opd.getDefiningOp();
        if (op_ == nullptr || isa<top::NoneOp>(op_)) {
          continue;
        }
        mm_ops.push_back(op_);
      }
      mm_ops.push_back(user);
      reshape_ops.push_back(user2);
    }
    mm_ops.append(reshape_ops);
    auto last_op = op;
    bool fixed = false;
    for (auto op_ : mm_ops) {
      if (last_op->getNextNode() != op_) {
        fixed = true;
        op_->moveAfter(last_op);
      }
      last_op = op_;
    }
    return fixed ? success() : failure();
  }
};

class OpReorderPass : public OpReorderBase<OpReorderPass> {
public:
  OpReorderPass() {}
  void runOnOperation() override {
    auto func = getOperation();
    if (func.getName() == "main") {
      // only for subnet
      return;
    }
    auto ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<OpReorderPattern>(ctx);
    applyPatternsAndFoldGreedily(func, std::move(patterns));
    // special for attention
    patterns.clear();
    patterns.add<AttentionReorderPattern>(ctx);
    applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

std::unique_ptr<OperationPass<FuncOp>> createOpReorderPass() {
  return std::make_unique<OpReorderPass>();
}
} // namespace tpu
} // namespace tpu_mlir

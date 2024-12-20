//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

// make sure operands is nearest to owner op
struct OpReorderPattern : public OpRewriterPatternEx3 {
  OpReorderPattern(MLIRContext *context)
      : OpRewriterPatternEx3(context, "OpReorderPattern", 1) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
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
  bool shouldPrint(Operation *op) const override { return false; }
};

struct AttentionReorderPattern : public OpRewriterPatternEx3 {
  AttentionReorderPattern(MLIRContext *context)
      : OpRewriterPatternEx3(context, "AttentionReorderPattern", 1) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
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
      auto user2 = *user->user_begin();
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
        } else if (!isa<top::WeightOp>(op_)) {
          return failure(); // move activations may cause use before define
                            // problem
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
  bool shouldPrint(Operation *op) const override { return false; }
};

static bool isLgSupport(Operation *op) {
  bool res = false;
  if (isa<top::WeightOp>(op)) {
    res = true;
  }
  if (auto lg_op = dyn_cast<tpu_mlir::LocalGenInterface>(op)) {
    res = mlir::succeeded(lg_op.LocalGenSupport());
  }
  return res;
}

struct GlobalOpReorderPattern : public OpRewriterPatternEx3 {
  GlobalOpReorderPattern(MLIRContext *context)
      : OpRewriterPatternEx3(context, "GlobalOpReorderPattern", 1) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const override {
    if (!isa<ReturnOp>(op)) {
      return failure();
    }
    auto opds = op->getOperands();
    for (auto opd : opds) {
      auto prev_op = opd.getDefiningOp();
      while (prev_op && !isLgSupport(prev_op)) {
        auto prev_opds = prev_op->getOperands();
        int num_act = 0;
        for (auto prev_opd : prev_opds) {
          auto pp_op = prev_opd.getDefiningOp();
          if (pp_op && !isa<FuncOp, top::WeightOp, top::NoneOp>(pp_op)) {
            num_act++;
          }
        }
        if (!prev_op->hasOneUse() || num_act > 1) {
          break;
        }
        prev_op->moveBefore(op);
        op = prev_op;
        prev_op = prev_op->getOperand(0).getDefiningOp();
      }
    }
    return success();
  }
  bool shouldPrint(Operation *op) const override { return false; }
};

struct ReshapeReorderPattern : public OpRewriterPatternEx3 {
  ReshapeReorderPattern(MLIRContext *context)
      : OpRewriterPatternEx3(context, "ReshapeReorderPattern", 1) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const override {
    if (!isa<tpu::ReshapeOp>(op) || op->getNumOperands() > 1)
      return failure();
    auto reshapeOp = cast<tpu::ReshapeOp>(op);
    auto parent_op = reshapeOp->getParentOp();
    assert(isa<FuncOp>(parent_op));
    auto funcOp = cast<FuncOp>(parent_op);

    // if reshape is directly connected to ReturnOp
    if ((dyn_cast_if_present<ReturnOp>(op->getNextNode()) ||
         dyn_cast_if_present<tpu::ReshapeOp>(op->getNextNode())) &&
        dyn_cast_if_present<top::InputOp>(op->getOperand(0).getDefiningOp())) {
      return failure();
    }

    auto &entryBlock = funcOp.getBody().back();
    mlir::Operation *returnOp = nullptr;
    for (auto &iter_op : entryBlock) {
      if (isa<ReturnOp>(iter_op)) {
        returnOp = &iter_op;
        break;
      }
    }
    if (returnOp) {
      for (auto operand : returnOp->getOperands()) {
        auto definingOp = operand.getDefiningOp();
        if (definingOp && isa<ReshapeOp>(definingOp) &&
            definingOp->hasOneUse()) {
          definingOp->moveBefore(returnOp);
          return success();
        }
      }
    }

    // if reshape is directly connected to function arguments,
    // this happens when reshapeOp is in subfunctions
    for (auto arg : funcOp.getArguments()) {
      if (dyn_cast_if_present<tpu::ReshapeOp>(op->getPrevNode()) &&
          arg == op->getOperand(0)) {
        return failure();
      }

      if (arg == op->getOperand(0)) {
        auto &entryBlock = funcOp.getBody().front();
        op->moveBefore(&entryBlock, entryBlock.begin());
        return success();
      }
    }

    // if op is in outer function, and is directly connected to top::InputOp
    if ((dyn_cast_if_present<top::InputOp>(op->getPrevNode()) ||
         dyn_cast_if_present<tpu::ReshapeOp>(op->getPrevNode())) &&
        dyn_cast_if_present<top::InputOp>(op->getOperand(0).getDefiningOp())) {
      return failure();
    }

    if (auto inputOp =
            dyn_cast<top::InputOp>(op->getOperand(0).getDefiningOp())) {
      op->moveAfter(inputOp.getOperation());
      return success();
    }
    return failure();
  }
  bool shouldPrint(Operation *op) const override { return false; }
};

class OpReorderPass : public OpReorderBase<OpReorderPass> {
public:
  OpReorderPass() {}
  void runOnOperation() override {
    auto ctx = &getContext();
    auto modules = module::getAllModules();
    for (auto s : *modules) {
      for (auto func : s.getOps<FuncOp>()) {
        if (func.getName() == "main") {
          // only for subnet
          continue;
        }
        RewritePatternSet patterns(ctx);
        patterns.add<OpReorderPattern>(ctx);
        // applyPatternsAndFoldGreedily(func, std::move(patterns));
        // special for attention
        patterns.clear();
        patterns.add<ReshapeReorderPattern>(ctx);
        applyPatternsAndFoldGreedily(func, std::move(patterns));

        // This pattern will lead to negative optimization, so disable it until
        // an update in the future
        patterns.clear();
        patterns.add<AttentionReorderPattern>(ctx);
        applyPatternsAndFoldGreedily(func, std::move(patterns));

        patterns.clear();
        patterns.add<GlobalOpReorderPattern>(ctx);
        applyPatternsAndFoldGreedily(func, std::move(patterns));
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createOpReorderPass() {
  return std::make_unique<OpReorderPass>();
}
} // namespace tpu
} // namespace tpu_mlir

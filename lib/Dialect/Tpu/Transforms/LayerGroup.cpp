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
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupOps.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <set>
#include <sstream>

using namespace llvm;
using namespace mlir;

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
        // if (op_->hasOneUse() == false) {
        //   op_->dump();
        //   llvm_unreachable("weightOp should only has one use");
        // }
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

class LayerGroupPass : public LayerGroupBase<LayerGroupPass> {
public:
  LayerGroupPass() {}
  void runOnOperation() override {
    auto func = getOperation();
    if (func.getName() == "main" || getRunMode(func) != RunMode::TPU_STATIC) {
      return;
    }
    int64_t opt = this->opt;
    auto ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<OpReorderPattern>(ctx);
    applyPatternsAndFoldGreedily(func, std::move(patterns));
    GroupOps gOps(func);
    gOps.process(opt);
  }
};

std::unique_ptr<OperationPass<FuncOp>> createLayerGroupPass() {
  return std::make_unique<LayerGroupPass>();
}
} // namespace tpu
} // namespace tpu_mlir

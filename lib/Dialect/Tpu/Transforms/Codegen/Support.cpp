//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Support.h"

namespace tpu_mlir {
namespace tpu {

class UnpackCoreParallelPattern
    : public OpRewriterPatternEx<tpu::CoreParallelOp> {
public:
  UnpackCoreParallelPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::CoreParallelOp>(context,
                                                 "UnpackCoreParallelPattern") {}
  LogicalResult matchAndRewriteImpl(tpu::CoreParallelOp op,
                                    PatternRewriter &rewriter) const override;
  bool shouldPrint(tpu::CoreParallelOp op) const override { return false; }
};

class UnpackGroupParallelPattern
    : public OpRewriterPatternEx<tpu::GroupParallelOp> {
public:
  UnpackGroupParallelPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::GroupParallelOp>(
            context, "UnpackGroupParallelPattern") {}
  LogicalResult matchAndRewriteImpl(tpu::GroupParallelOp op,
                                    PatternRewriter &rewriter) const override;
  bool shouldPrint(tpu::GroupParallelOp op) const override { return false; }
};

LogicalResult UnpackCoreParallelPattern::matchAndRewriteImpl(
    tpu::CoreParallelOp op, PatternRewriter &rewriter) const {
  Operation *last_op = nullptr;
  // set core id first
  auto prev_op = op.getBody().front().getTerminator()->getPrevNode();
  auto join_op = dyn_cast_or_null<tpu::CoreJoinOp>(prev_op);
  if (!join_op) {
    UNREACHABLE_OP("CoreJoinOp not found in CoreParallelOp!", op);
  }
  int num_inputs = join_op.getNumOperands();
  assert(num_inputs == module::getCoreNum());
  for (int i = 0; i < num_inputs; i++) {
    auto in_op = join_op.getOperand(i).getDefiningOp();
    if (!in_op) {
      UNREACHABLE_OP("Defining op not found for CoreJoinOp input!", op);
    }
    in_op->setAttr(CodegenAttr::CORE_ID, rewriter.getI32IntegerAttr(i));
  }
  // unpack core parallel region
  while (op.getBody().front().empty() == false) {
    auto &inner_op = op.getBody().front().front();
    if (isa<tpu::YieldOp>(inner_op)) {
      break;
    }
    if (auto split_op = dyn_cast<tpu::CoreSplitOp>(inner_op)) {
      if (split_op.getNumResults() == 1) {
        auto out = split_op.getResult(0);
        auto in = split_op.getInput();
        rewriter.replaceAllUsesWith(out, in);
        rewriter.eraseOp(&inner_op);
        continue;
      }
    }
    rewriter.setInsertionPoint(op);
    auto new_op = rewriter.clone(inner_op);
    if (isa<tpu::CoreJoinOp>(inner_op)) {
      last_op = new_op;
      break;
    }
    rewriter.replaceOp(&inner_op, new_op);
  }
  if (last_op == nullptr) {
    UNREACHABLE_OP("UnpackCoreParallelPattern failed!!", op);
  }
  rewriter.replaceOp(op, last_op);
  return success();
}

LogicalResult UnpackGroupParallelPattern::matchAndRewriteImpl(
    tpu::GroupParallelOp op, PatternRewriter &rewriter) const {
  Operation *last_op = nullptr;
  auto region_num = op.getRegions().size();
  assert(region_num == module::getCoreNum());
  std::vector<Value> results;
  bool first = true;
  for (int i = 0; i < region_num; i++) {
    int core_id = i;
    auto &block = op.getRegion(i).front();
    while (block.empty() == false) {
      auto &inner_op = block.front();
      if (isa<tpu::YieldOp>(inner_op)) {
        break;
      }
      rewriter.setInsertionPoint(op);
      auto new_op = rewriter.clone(inner_op);
      new_op->setAttr(CodegenAttr::CORE_ID,
                      rewriter.getI32IntegerAttr(core_id));
      if (first) {
        first = false;
        new_op->setAttr(CodegenAttr::SYNC_ALL_BEGIN,
                        rewriter.getBoolAttr(true));
      }
      rewriter.replaceOp(&inner_op, new_op);
      last_op = new_op;
    }
    if (last_op == nullptr) {
      UNREACHABLE_OP("UnpackGroupParallelPattern failed!!", op);
    }
    if (i == region_num - 1) {
      last_op->setAttr(CodegenAttr::SYNC_ALL_END, rewriter.getBoolAttr(true));
    }
    results.insert(results.end(), last_op->getResults().begin(),
                   last_op->getResults().end());
  }
  rewriter.replaceOp(op, results);
  return success();
}

void DoPatternsForDynamic(ModuleOp m) {
  for (auto func : m.getOps<func::FuncOp>()) {
    auto rmode = getRunMode(func);
    if (rmode != RunMode::TPU_DYNAMIC) {
      continue;
    }
    RewritePatternSet patterns(m.getContext());
    patterns.insert<UnpackCoreParallelPattern>(m.getContext());
    patterns.insert<UnpackGroupParallelPattern>(m.getContext());
    applyPatternsAndFoldGreedily(func, std::move(patterns));
    // set flags for multi core
    func.walk([&](Operation *op) {
      if (module::isOpInBlock(op)) {
        return;
      }
      if (auto core_split = dyn_cast<tpu::CoreSplitOp>(op)) {
        auto former_op = core_split->getPrevNode();
        if (!isa_and_nonnull<tpu::CoreSplitOp>(former_op)) {
          // avoid CoreSplit + CoreSplit case
          core_split->setAttr(CodegenAttr::SYNC_ALL_BEGIN,
                              BoolAttr::get(op->getContext(), true));
        }
      } else if (auto core_join = dyn_cast<tpu::CoreJoinOp>(op)) {
        core_join->setAttr(CodegenAttr::SYNC_ALL_END,
                           BoolAttr::get(op->getContext(), true));
        uint32_t num_operands = core_join.getNumOperands();
        for (uint32_t i = 0; i < num_operands; i++) {
          auto in_op = core_join.getOperand(i).getDefiningOp();
          if (i == 0) {
            in_op->setAttr(CodegenAttr::ADDR_JOIN_START,
                           BoolAttr::get(op->getContext(), true));
          } else {
            in_op->setAttr(CodegenAttr::ADDR_JOIN_NEXT,
                           BoolAttr::get(op->getContext(), true));
          }
        }
      } else if (auto gop = dyn_cast<tpu::GroupOp>(op)) {
        auto secs = gop.getNsecs() * gop.getCsecs() * gop.getHsecs() *
                    gop.getWsecs() * gop.getDsecs();
        if (secs < 2) {
          return;
        }
        op->setAttr(CodegenAttr::SYNC_ALL_END,
                    BoolAttr::get(op->getContext(), true));
        auto former_op = gop->getPrevNode();
        while (isa_and_nonnull<top::WeightOp, top::NoneOp>(former_op)) {
          former_op = former_op->getPrevNode();
        }
        if (former_op) {
          // TODO: LayerGroup(1 slice) + LayerGroup(N slice) will fail sync all
          // Need to be fixed in the future
          former_op->setAttr(CodegenAttr::SYNC_ALL_END,
                             BoolAttr::get(op->getContext(), true));
        }
      } else if (op->hasAttrOfType<BoolAttr>("multicore")) {
        op->setAttr(CodegenAttr::SYNC_ALL_BEGIN,
                    BoolAttr::get(op->getContext(), true));
        op->setAttr(CodegenAttr::SYNC_ALL_END,
                    BoolAttr::get(op->getContext(), true));
      }
    });
  }
  // remove redundant sync_all
  for (auto func : m.getOps<func::FuncOp>()) {
    auto rmode = getRunMode(func);
    if (rmode != RunMode::TPU_DYNAMIC) {
      continue;
    }
    for (auto iter = func.getBody().front().begin();
         iter != func.getBody().front().end(); ++iter) {
      auto op = &*iter;
      if (module::isOpInBlock(op)) {
        continue;
      }
      if (!op->hasAttrOfType<BoolAttr>(CodegenAttr::SYNC_ALL_END)) {
        continue;
      }
      auto next_op = op->getNextNode();
      while (isa_and_nonnull<top::WeightOp>(next_op)) {
        next_op = next_op->getNextNode();
      }
      if (next_op == nullptr) {
        break;
      }
      if (next_op->hasAttrOfType<BoolAttr>(CodegenAttr::SYNC_ALL_BEGIN)) {
        next_op->removeAttr(CodegenAttr::SYNC_ALL_BEGIN);
      }
    }
  }
}

} // namespace tpu
} // namespace tpu_mlir

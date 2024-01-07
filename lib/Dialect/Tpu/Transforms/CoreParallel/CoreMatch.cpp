#include "CoreParallel.hpp"

namespace tpu_mlir {
namespace tpu {

static bool isOpSameCalc(Operation *op0, Operation *op1) {
  auto compare = [&](mlir::ValueRange left, mlir::ValueRange right) -> bool {
    for (auto it : llvm::zip(left, right)) {
      auto left = std::get<0>(it);
      auto right = std::get<1>(it);
      if (module::isNone(left) || module::isNone(right)) {
        continue;
      }
      auto l_s = module::getShape(left);
      auto r_s = module::getShape(right);
      if (l_s != r_s) {
        return false;
      }
    }
    return true;
  };
  if (op0 == op1) {
    // can't be the same op
    return false;
  }
  if (op0->getName() != op1->getName()) {
    return false;
  }
  if (false == compare(op0->getOperands(), op1->getOperands())) {
    return false;
  }
  if (false == compare(op0->getResults(), op1->getResults())) {
    return false;
  }
  return true;
}

static bool isOpSameCalc(const std::vector<Operation *> &ops) {
  if (ops.size() < 2) {
    return false;
  }
  for (int i = 1; i < ops.size(); i++) {
    if (!isOpSameCalc(ops[0], ops[i])) {
      return false;
    }
  }
  return true;
}

static void core_match(PatternRewriter &rewriter,
                       std::vector<Operation *> ops_begin,
                       std::vector<Operation *> ops_end,
                       tpu::CorePattern pattern) {
  // 1. Create pattern params
  auto ctx = rewriter.getContext();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "pattern", tpu::CorePatternAttr::get(ctx, pattern)));

  // 2. Insert CoreBeginOp
  std::vector<Value> inputs;
  std::vector<Type> types;
  std::vector<Location> locs;
  Value opd = ops_begin[0]->getOperand(0);
  for (auto op : ops_begin) {
    auto input = op->getOperand(0);
    auto type = input.getType();
    auto loc = module::getLocLike(op, "core_begin");
    inputs.push_back(input);
    types.push_back(type);
    locs.push_back(loc);
  }
  auto begin_loc = FusedLoc::get(ctx, locs);
  rewriter.setInsertionPointAfterValue(opd);
  auto begin =
      rewriter.create<tpu::CoreBeginOp>(begin_loc, types, inputs, attrs);
  for (auto op : ops_begin) {
    if (op != ops_begin[0]) {
      auto in = op->getOperand(0);
      if (!isa<BlockArgument>(in)) {
        in.getDefiningOp()->moveBefore(begin);
      }
    }
  }
  for (size_t i = 0; i < ops_begin.size(); ++i) {
    int index = 0;
    for (auto [idx, in] : llvm::enumerate(ops_begin[i]->getOperands())) {
      if (in == begin.getInputs()[i]) {
        index = idx;
        break;
      }
    }
    ops_begin[i]->setOperand(index, begin.getOutputs()[i]);
  }

  // 3. Insert CoreEndOp
  attrs.clear();
  inputs.clear();
  types.clear();
  locs.clear();

  attrs.push_back(rewriter.getNamedAttr(
      "pattern", tpu::CorePatternAttr::get(ctx, pattern)));

  for (auto op : ops_end) {
    for (auto o : op->getResults()) {
      inputs.push_back(o);
      types.push_back(o.getType());
      auto loc = module::getLocLike(op, "core_end");
      locs.push_back(loc);
      break;
    }
  }
  auto end_loc = FusedLoc::get(ctx, locs);
  rewriter.setInsertionPointAfter(ops_end[0]);
  auto end = rewriter.create<tpu::CoreEndOp>(end_loc, types, inputs, attrs);

  for (size_t i = 0; i < ops_end.size(); ++i) {
    inputs[i].replaceUsesWithIf(end.getOutputs()[i], [&](OpOperand &use) {
      return use.getOwner() != end;
    });
  }
}

static void common_match(PatternRewriter &rewriter,
                         std::vector<Operation *> &ops) {
  std::vector<Operation *> ops_begin = ops;
  std::vector<Operation *> ops_end = ops;
  auto num_ops = ops_begin.size();
  bool next_is_same = true;
  do {
    std::vector<Operation *> next_ops;
    for (int i = 0; i < num_ops; i++) {
      auto op = ops_end[i];
      if (!op->hasOneUse()) {
        break;
      }
      auto next_op = *op->getUsers().begin();
      if (isa<ReturnOp>(next_op)) {
        break;
      }
      next_ops.push_back(next_op);
    }
    if (next_ops.size() != num_ops || !isOpSameCalc(next_ops)) {
      next_is_same = false;
      continue;
    }
    ops_end = next_ops;
  } while (next_is_same);
  core_match(rewriter, ops_begin, ops_end, tpu::CorePattern::Common);
}

// if operations are the same, then run in multi cores
struct CommonMatch : public RewritePattern {
  CommonMatch(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (module::isOpInCoreMatch(op) || module::isOpInGroup(op)) {
      return failure();
    }
    if (isa<ReturnOp, FuncOp, tpu::YieldOp, top::YieldOp, top::WeightOp>(op)) {
      return failure();
    }
    if (op->getNumResults() != 1) {
      return failure();
    }
    auto num_users = std::distance(op->user_begin(), op->user_end());
    if (num_users < 2) {
      return failure();
    }
    auto find_f = [&](std::vector<std::vector<Operation *>> &ops,
                      Operation *op) -> bool {
      for (auto v : ops) {
        for (auto op_ : v) {
          if (op_ == op) {
            return true;
          }
        }
      }
      return false;
    };
    auto users = op->getUsers();
    std::vector<std::vector<Operation *>> same_ops;
    for (auto left = users.begin(); left != users.end(); left++) {
      auto left_op = *left;
      if (isa<tpu::ReshapeOp>(left_op)) {
        continue;
      }
      if (find_f(same_ops, left_op)) {
        continue;
      }
      std::vector<Operation *> ops = {left_op};
      auto right = left;
      for (right++; right != users.end(); right++) {
        auto right_op = *right;
        if (find_f(same_ops, right_op)) {
          continue;
        }
        if (isOpSameCalc(left_op, right_op)) {
          ops.push_back(right_op);
        }
      }
      if (ops.size() > 1) {
        same_ops.emplace_back(ops);
      }
    }
    if (same_ops.empty()) {
      return failure();
    }
    for (auto ops : same_ops) {
      common_match(rewriter, ops);
    }
    return success();
  }
};

void doCoreParallelPattern(ModuleOp m) {
  // first match pattern
  // module::applyPatternOnce<CommonMatch>(m);
  // then split different pattern to multi cores
  //....
}

} // namespace tpu
} // namespace tpu_mlir

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/CoreParallel/CoreParallel.hpp"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include <llvm/ADT/DenseSet.h>
#include <unordered_set>
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

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

static Operation *cloneOp(PatternRewriter &rewriter, Operation *op,
                          llvm::ArrayRef<int64_t> new_shape,
                          llvm::StringRef suffix) {
  rewriter.setInsertionPointAfter(op);
  auto new_op = rewriter.clone(*op);
  for (auto r : new_op->getResults()) {
    module::setShape(r, new_shape);
  }
  module::setLocSuffix(new_op, suffix);
  return new_op;
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
      locs.emplace_back(loc);
      break;
    }
  }
  auto endLocs = FusedLoc::get(ctx, locs);
  rewriter.setInsertionPointAfter(ops_end[0]);
  auto end = rewriter.create<tpu::CoreEndOp>(endLocs, types, inputs, attrs);

  for (size_t i = 0; i < ops_end.size(); ++i) {
    inputs[i].replaceUsesWithIf(end.getOutputs()[i], [&](OpOperand &use) {
      return use.getOwner() != end;
    });
  }
}

// check if there is a dataFlow from a to b.
bool isReachable(Operation *a, Operation *b) {
  if (a && a == b)
    return true;
  if (b->getNumOperands() == 0)
    return false;
  if (a->getBlock() == b->getBlock()) {
    if (!a->isBeforeInBlock(b))
      return false;
  }
  if (!isa<FuncOp>(a->getParentOp()) || isa<tpu::YieldOp, top::YieldOp>(a))
    return isReachable(a->getParentOp(), b);
  return llvm::any_of(a->getUsers(),
                      [b](Operation *op) { return isReachable(op, b); });
}

bool isReachable_for_training(Operation *a, Operation *b, std::unordered_set<Operation *> &visited) {
  if (a && a == b)
    return true;
  if (b->getNumOperands() == 0)
    return false;
  if (visited.count(a))
    return false;
  visited.insert(a);
  if (a->getBlock() == b->getBlock()) {
    if (!a->isBeforeInBlock(b))
      return false;
  }
  if (!isa<FuncOp>(a->getParentOp()) || isa<tpu::YieldOp, top::YieldOp>(a))
    return isReachable_for_training(a->getParentOp(), b, visited);
  return llvm::any_of(a->getUsers(),
                      [b, &visited](Operation *op) { return isReachable_for_training(op, b, visited); });
}


bool isCircularDependency(std::vector<Operation *> &beginOps,
                          std::vector<Operation *> &endOps) {
  // Operations use value in [beginOps, endOps]
  llvm::DenseSet<Operation *> usedByOutside;
  // Operations used by [beginOps, endOps]
  llvm::DenseSet<Operation *> definedInOutside;
  llvm::DenseSet<Operation *> blockOps;

  // All operations in usedByOutside should not dominate those in
  // definedInOutside.
  for (auto [beginOp, endOp] : llvm::zip(beginOps, endOps)) {
    Operation *it = beginOp;
    for (;; it = *it->getUsers().begin()) {
      for (auto v : it->getOperands()) {
        if (auto op = v.getDefiningOp()) {
          definedInOutside.insert(op);
        }
      }
      for (auto op : it->getUsers()) {
        if (op) {
          usedByOutside.insert(op);
        }
      }
      blockOps.insert(it);
      if (it == endOp) {
        break;
      }
    }
  }
  for (auto v : blockOps) {
    usedByOutside.erase(v);
    definedInOutside.erase(v);
  }
  if (module::isTrain()){
    std::unordered_set<Operation *> visited;
    // check dataFlow form usedByOutside to definedInOutside.
    for (auto uOp : usedByOutside) {
      for (auto dOp : definedInOutside) {
        if (isReachable_for_training(uOp, dOp, visited)) {
          return true;
        }
      }
    }
  }
  else{
    for (auto uOp : usedByOutside) {
      for (auto dOp : definedInOutside) {
        if (isReachable(uOp, dOp)) {
          return true;
        }
      }
    }
  }
  return false;
}
bool checkDataDependencies(const std::vector<Operation *> &ops) {
  bool hasDependency = false;
  for (Operation *consumer : ops) {

    for (auto operand : consumer->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      if (producer &&
          std::find(ops.begin(), ops.end(), producer) != ops.end()) {
        hasDependency = true;
      }
    }
  }
  return hasDependency;
}

bool checkForReturnOpUser(const std::vector<Operation *> &ops) {
  for (Operation *op : ops) {
    for (auto *user : op->getUsers()) {
      if (isa<ReturnOp>(user)) {
        return true;
      }
    }
  }
  return false;
}
static void group_distribute(PatternRewriter &rewriter,
                             std::vector<Operation *> beginOps,
                             std::vector<Operation *> endOps,
                             tpu::CorePattern pattern) {
  auto ctx = rewriter.getContext();
  std::vector<Type> returnTypes;
  std::vector<Location> locs;
  SmallVector<Value> endValues;

  if (isCircularDependency(beginOps, endOps))
    return;
  if (checkDataDependencies(endOps))
    return;
  for (auto op : endOps) {
    for (auto o : op->getResults()) {
      endValues.push_back(o);
      returnTypes.push_back(o.getType());
      locs.emplace_back(op->getLoc());
      break;
    }
  }

  auto endLocs = FusedLoc::get(ctx, locs);
  auto theEndOp = endOps[0];
  for (int i = 1, n = endOps.size(); i < n; i++) {
    if (theEndOp->getBlock() != endOps[i]->getBlock()) {
      // This should be unreachable! FIX ME.
      // does not supports op in different blocks.
      return;
    }
    if (theEndOp->isBeforeInBlock(endOps[i]))
      theEndOp = endOps[i];
  }

  rewriter.setInsertionPointAfter(theEndOp);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "pattern", tpu::CorePatternAttr::get(ctx, pattern)));
  auto groupParallelOp = rewriter.create<tpu::GroupParallelOp>(
      endLocs, returnTypes, ValueRange{}, attrs, beginOps.size());

  llvm::SetVector<Value> groupParallelOperands;

  for (auto [v, r] : llvm::zip(endValues, groupParallelOp.getOutputs())) {
    v.replaceAllUsesWith(r);
  }

  for (auto [opBegin, opEnd, subgraph] :
       llvm::zip(beginOps, endOps, groupParallelOp.getParallel())) {
    auto it = opBegin;
    auto body = new Block();
    subgraph.push_back(body);
    rewriter.setInsertionPointToStart(body);
    auto yieldOp =
        rewriter.create<tpu::YieldOp>(opEnd->getLoc(), opEnd->getResults());
    opEnd->moveBefore(yieldOp);
    for (; it != opEnd; it = *it->getUsers().begin()) {
      it->moveBefore(opEnd);
    }
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
      // Disable the in-place operation since it complicates the process of
      // address assignment. TODO: refine address assignment.
      if (isa<ReturnOp, tpu::ReshapeOp, tpu::ConcatOp, tpu::IdentityOp>(
              next_op)) {
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

  DEBUG_WITH_TYPE("group-parallel", {
    llvm::dbgs() << "; action = group-parallel"
                << "; stage = do_group_distribute"
                << "\n";
  });
  group_distribute(rewriter, ops_begin, ops_end, tpu::CorePattern::Common);
}

struct FuncInputMatch : public OpRewriterPatternEx<FuncOp> {
  FuncInputMatch(MLIRContext *context)
      : OpRewriterPatternEx<FuncOp>(context, "FuncInputMatch", 1) {}
  LogicalResult matchAndRewriteImpl(FuncOp func,
                                    PatternRewriter &rewriter) const override {
    auto num_core = module::getCoreNum();
    if (module::getName(func) == "main") {
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

    DEBUG_WITH_TYPE("group-parallel", {
      llvm::dbgs() << "; action = group-parallel" <<
                    "; func = " << module::getName(func) <<
                    "; num_core = " << num_core <<
      "\n";
    });

    std::vector<std::vector<Operation *>> same_ops;
    auto prepareSameOps = [&](BlockArgument arg) {
      auto users = arg.getUsers();
      for (auto left = users.begin(); left != users.end(); left++) {
        auto left_op = *left;
        // inPlace op
        DEBUG_WITH_TYPE("group-parallel", {
          llvm::dbgs() << "; action = group-parallel"
                      << "; stage = prepareSameOps"
                      << "; step = check_op"
                      << "; op = " << module::getName(left_op)
                      << "\n";
        });
        if (isa<tpu::ReshapeOp, tpu::SliceOp, tpu::ConcatOp, tpu::GroupOp>(
                left_op)) {
          DEBUG_WITH_TYPE("group-parallel", {
            llvm::dbgs() << "; action = group-parallel"
                          << "; stage = prepareSameOps"
                        << "; step = continue"
                        << "; op = " << module::getName(left_op)
                        << "; reason = meetInvalidOp" << "\n";
          });
          continue;
        }
        if (module::isOpInBlock(left_op)) {
          DEBUG_WITH_TYPE("group-parallel", {
            llvm::dbgs() << "; action = group-parallel"
                        << "; stage = prepareSameOps"
                        << "; step = continue"
                        << "; op = " << module::getName(left_op)
                        << "; reason = OpInBlock" << "\n";
          });
          continue;
        }
        if (find_f(same_ops, left_op)) {
          DEBUG_WITH_TYPE("group-parallel", {
            llvm::dbgs() << "; action = group-parallel"
                        << "; stage = prepareSameOps"
                        << "; step = continue"
                        << "; op = " << module::getName(left_op)
                        << "; reason = sameOp" << "\n";
          });
          continue;
        }
        std::vector<Operation *> ops = {left_op};
        auto right = left;
        for (right++; right != users.end(); right++) {
          auto right_op = *right;
          if (find_f(same_ops, right_op)) {
            continue;
          }
          if (module::isOpInBlock(right_op)) {
            continue;
          }
          if (isOpSameCalc(left_op, right_op)) {
            ops.push_back(right_op);
          }
          if (ops.size() == num_core) {
            break;
          }
        }
        if (ops.size() == num_core) {
          same_ops.emplace_back(ops);
        }
      }

      if (same_ops.empty()) {
        DEBUG_WITH_TYPE("group-parallel", {
          llvm::dbgs() << "; action = group-parallel"
                      << "; stage = prepareSameOps"
                      << "; step = failure"
                      << "; reason = noSameOp" << "\n";
          arg.dump();
        });
        return failure();
      }
      for (auto ops : same_ops) {
        if (checkDataDependencies(ops) || checkForReturnOpUser(ops)) {
          DEBUG_WITH_TYPE("group-parallel", {
            llvm::dbgs() << "; action = group-parallel"
                        << "; stage = prepareSameOps"
                        << "; step = failure"
                        << "; reason = hasDependency" << "\n";
          });
          return failure();
        }
      }
      DEBUG_WITH_TYPE("group-parallel", {
        llvm::dbgs() << "; action = group-parallel"
                     << "; stage = " << "do_common_match"
                     << "\n";
        arg.dump();
      });
      for (auto ops : same_ops) {
        common_match(rewriter, ops);
      }
      return success();
    };

    for (auto it : llvm::enumerate(func.getArguments())) {
      auto arg = it.value();
      auto num_users = std::distance(arg.user_begin(), arg.user_end());
      DEBUG_WITH_TYPE("group-parallel", {
        llvm::dbgs() << "; action = group-parallel"
                     << "; arg_index = " << it.index()
                     << "; num_users = " << num_users << "\n";
        arg.dump();
      });
      if (num_users < num_core) {
        continue;
      }
      prepareSameOps(arg);
    }

    return success();
  }
};

// if operations are the same, then run in multi cores
struct CommonMatch : public OpRewriterPatternEx3 {
  CommonMatch(MLIRContext *context)
      : OpRewriterPatternEx3(context,"CommonMatch",1) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto num_core = module::getCoreNum();

    if (!isa<FuncOp>(op->getParentOp())) {
      return failure();
    }
    if (isa<ReturnOp, top::NoneOp, FuncOp, tpu::YieldOp, top::YieldOp,
            top::WeightOp>(op)) {
      return failure();
    }

    auto num_users = std::distance(op->user_begin(), op->user_end());
    if (num_users < num_core) {
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

    std::vector<std::vector<Operation *>> same_ops;
    for (auto value : op->getResults()) {
      auto users = value.getUsers();
      for (auto left = users.begin(); left != users.end(); left++) {
        auto left_op = *left;
        // inPlace op
        if (isa<tpu::ReshapeOp, tpu::SliceOp, tpu::ConcatOp, tpu::GroupOp>(left_op)) {
          continue;
        }
        if (module::isOpInBlock(left_op)) {
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
          if (module::isOpInBlock(right_op)) {
            continue;
          }
          if (isOpSameCalc(left_op, right_op)) {
            ops.push_back(right_op);
          }
          if (ops.size() == num_core) {
            break;
          }
        }
        if (ops.size() == num_core) {
          same_ops.emplace_back(ops);
        }
      }
    }

    if (same_ops.empty()) {
      return failure();
    }
    for (auto ops : same_ops) {
      if (checkDataDependencies(ops) || checkForReturnOpUser(ops)) {
        return failure();
      }
    }
    for (auto ops : same_ops) {
      common_match(rewriter, ops);
    }
    return success();
  }
  bool shouldPrint(Operation *op) const override { return false;}
};

class A16MatMulMatch  : public OpRewriterPatternEx<tpu::A16MatMulOp> {
  public:
  A16MatMulMatch(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::A16MatMulOp>(context,"A16MatMulMatch") {}

  LogicalResult matchAndRewriteImpl(tpu::A16MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto num_cores = module::getCoreNum();
    if (supportMultiCore(op)) {
      return failure();
    }
    if (op.getWeightBits() != 4) {
      return failure();
    }
    if (module::isOpInBlock(op)) {
      return failure();
    }
    if (!op.getWTranspose()) {
      UNREACHABLE_OP("Not Implemented", op);
    }
    auto w = op.getWeight();
    auto s = op.getScale();
    auto zp = op.getZp();
    auto bias = op.getBias();
    if (module::isActive(bias)) {
      UNREACHABLE_OP("Not Implemented", op);
    }
    auto out = op.getOutput();
    auto w_shape = module::getShape(w);
    auto s_shape = module::getShape(s);
    auto o_shape = module::getShape(out);
    auto N = w_shape[0];
    auto G = s_shape[1];
    if (N % num_cores != 0 || G % num_cores != 0) {
      UNREACHABLE_OP("Not Implemented", op);
    }
    auto N_slice = N / num_cores;
    auto G_slice = G / num_cores;
    std::vector<Operation *> ops_begin;
    std::vector<Operation *> ops_end;
    std::vector<Value> concat_operands;
    for (int i = 0; i < num_cores; i++) {
      auto suffix = std::to_string(i);
      auto new_w = module::opSliceAxis(rewriter, w, 0, i * N_slice, N_slice);
      auto new_s = module::opSliceAxis(rewriter, s, 1, i * G_slice, G_slice);
      auto new_zp = module::opSliceAxis(rewriter, zp, 1, i * G_slice, G_slice);
      std::vector<int64_t> shape = o_shape;
      shape.back() = N_slice;
      auto new_op = cloneOp(rewriter, op, shape, suffix);
      new_op->setOperand(1, new_w);
      new_op->setOperand(2, new_s);
      new_op->setOperand(3, new_zp);
      if (module::isWeight(bias)) {
        auto b_shape = module::getShape(bias);
        auto new_b = module::opSliceAxis(rewriter, bias, b_shape.size() - 1,
                                         i * N_slice, N_slice);
        new_op->setOperand(4, new_b);
      }
      concat_operands.push_back(new_op->getResult(0));
      ops_begin.push_back(new_op);
      ops_end.push_back(new_op);
    }
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(rewriter.getNamedAttr(
        "axis", rewriter.getSI32IntegerAttr(o_shape.size() - 1)));
    rewriter.replaceOpWithNewOp<tpu::ConcatOp>(op, out.getType(),
                                               concat_operands, attrs);
    group_distribute(rewriter, ops_begin, ops_end, tpu::CorePattern::Common);
    return success();
  }
  bool shouldPrint(tpu::A16MatMulOp op) const override { return false;}
};

#if 0
// test case: Gemma-2B block
// RMSNorm --->A16MatMul------------> Mul -> A16MatMul
//         --->A16MatMul--->Active /
class MlpA16Match  : public OpRewriterPatternEx<tpu::A16MatMulOp> {
  public:
  MlpA16Match(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::A16MatMulOp>(context,"MlpA16Match") {}

  bool is_support(tpu::A16MatMulOp op) {
    if (module::isOpInBlock(op)) {
      return false;
    }
    if (!module::isWeight(op.getRight())) {
      // TODO: support bias weight
      return false;
    }
    auto p = op.parseParam();
    if (p.batch != 1 || p.M != 1 || p.do_relu || p.with_bias) {
      // TODO: if do_relu or with_bias, need do relu after add bias
      return false;
    }
    return true;
  }

  LogicalResult matchAndRewriteImpl(tpu::A16MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto num_cores = module::getCoreNum();
    if (num_cores < 2) {
      return failure();
    }
    if (!is_support(op)) {
      return failure();
    }

    auto in_op = op.getInput().getDefiningOp();
    if (!in_op) {
      return failure();
    }
    if (!isa<tpu::MulOp, tpu::AddOp>(in_op)) {
      return failure();
    }
    if (in_op->getNumOperands() != 2) {
      return failure();
    }
    auto left_op = in_op->getOperands(0).getDefiningOp();
    auto right_op = in_op->getOperands(1).getDefiningOp();
    if (!isa<tpu::ActiveOp>(left_op) || !is_support(right_op)) {
      return failure();
    }
    auto left_in_op = left_op->getOperands(0);
    if (!is_support(left_in_op)) {
      return failure();
    }
    if (left_in_op->getOperands(0) != right_op->getOperands(0)) {
      return failure();
    }
    auto norm_op = left_in_op->getOperands(0).getDefiningOp();
    if (!isa<tpu::RMSNormOp, tpu::LayerNormOp>(norm_op)) {
      return failure();
    }
    // bingo !!!
    std::vector<Operation *> ops_begin;
    std::vector<Operation *> ops_end;
    for (int i = 0; i < num_cores; i++) {
      auto suffix = std::to_string(i);
      auto norm_shape = module::getShape(norm_shape);
      auto new_norm = cloneOp(rewriter, norm_op, norm_shape, suffix);
      new_norm.setOperands(norm_op.getOperands());
      ops_begin.push_back(new_norm);
      auto l_mm = cast<MMTy>(left_in_op);
      auto l_p = left_mm.parseParam();
      auto l_N_slice = ceiling_func(l_p.N, num_cores);
      auto l_offset = i * l_N_slice;
      auto l_slice = std::min(l_N_slice, l_p.N - l_offset);
      auto l_filter_shape = module::getShape(l_p.getRight());
      auto l_filter_dims = l_filter_sape.size();
      auto l_filter = module::opSliceAxis(l_p.getOperand(1), l_filter_dims - 1,
                                          l_offset, l_slice);
      if (is_a16) {

      } else {
      }
    }
    auto out = op.getOutput();
    auto in_shape = module::getShape(op.getInput());
    auto in_dims = in_shape.size();
    auto r_shape = module::getShape(op.getRight());
    auto r_dims = r_shape.size();
    auto new_K = p.K / num_cores;
    std::vector<Value> add_operands;
    for (int i = 0; i < num_cores; i++) {
      std::vector<Value> operands;
      auto newInput = module::opSliceAxis(rewriter, op.getInput(), in_dims - 1,
                                          i * new_K, new_K, true);
      operands.push_back(newInput);
      auto newRight = module::opSliceAxis(rewriter, op.getRight(), r_dims - 2,
                                          i * new_K, new_K, true);
      operands.push_back(newRight);
      operands.push_back(op.getBias());
      auto suffix = std::to_string(i);
      auto new_loc = module::getLocLike(out, suffix);
      rewriter.setInsertionPointAfterValue(newRight);
      auto newMM = rewriter.create<top::MatMulOp>(new_loc, out.getType(),
                                                  operands, op->getAttrs());
      add_operands.push_back(newMM.getOutput());
    }
    rewriter.replaceOpWithNewOp<top::AddOp>(op, out.getType(), add_operands);
    return success();
  }
  bool shouldPrint(tpu::A16MatMulOp op) const override { return false;}
};
#endif

void doCoreParallelPattern(ModuleOp m) {
  // first match pattern
  module::applyPatternOnce<CommonMatch>(m);
  module::applyPatternOnce<FuncInputMatch>(m);
  // then split different pattern to multi cores
  module::applyPatternOnce<A16MatMulMatch>(m);
  //....
  for (auto op : m.getOps<FuncOp>()) {
    for (auto &block : op.getBlocks()) {
      if (!mlir::sortTopologically(&block)) {
        llvm_unreachable(("This function '" + op.getName() +
                          "' contains circular dependencies.")
                             .str()
                             .c_str());
      };
    }
  }
}

} // namespace tpu
} // namespace tpu_mlir

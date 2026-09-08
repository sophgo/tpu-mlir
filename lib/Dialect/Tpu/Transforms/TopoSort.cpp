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
#include "tpu_mlir/Support/Module.h"

namespace tpu_mlir {
namespace tpu {

// static bool isLgSupport(Operation *op) {
//   bool res = false;
//   // if (isa<top::WeightOp>(op)) {
//   //   res = true;
//   // }
//   if (auto lg_op = dyn_cast<tpu_mlir::LocalGenInterface>(op)) {
//     res = mlir::succeeded(lg_op.LocalGenSupport());
//   }
//   return res;
// }

// static bool isReadyToSchedule(Operation *op, llvm::SetVector<Operation *>
// &unScheduledOps) {
//   if (!op || !unScheduledOps.contains(op))
//     return false;
//   for (auto in : op->getOperands()) {
//     auto in_op = in.getDefiningOp();
//     if (!in_op) {
//       continue;
//     }
//     if (unScheduledOps.contains(in_op)) {
//       return false;
//     }
//   }
//   return true;
// };

// static int getUnscheduledUserCount(Operation *op, llvm::SetVector<Operation
// *> &unScheduledOps) {
//   llvm::DenseSet<Operation*> unique_users;
//   for (auto out : op->getResults()) {
//     for (auto user : out.getUsers()) {
//       if (unScheduledOps.contains(user)) {
//         unique_users.insert(user);
//       }
//     }
//   }
//   return unique_users.size();
// };

// /**
//  * SORT STRANGIES:
//  * 1. depth first PARENT-CHILD-PARENT-CHILD-...
//  * 2. dominant node first for multi-branch structure
//  * 3. Keep the original order and sort
//  * NOTE: flag_local_turn for PING-PONG rule
//  *  */
// static bool tryDepthFirst(Operation *&nextOp, Operation *prevOp,
//                          llvm::SetVector<Operation *> &unScheduledOps,
//                          const bool flag_sort_local_ops) {
//   if (!prevOp)
//     return false;
//   if (prevOp->hasOneUse() == true) {
//     nextOp = *(prevOp->user_begin());
//     if (isReadyToSchedule(nextOp, unScheduledOps) &&
//         isLgSupport(nextOp) == flag_sort_local_ops) {
//       return true;
//     }
//   }
//   nextOp = nullptr;
//   return false;
// }

// static bool tryDominantNodeFirst(Operation *&nextOp,
// llvm::SetVector<Operation *> &unScheduledOps,
//                                  llvm::SetVector<Operation *> &dominantOps,
//                                  const bool flag_local_turn) {
//   dominantOps.remove_if([&](Operation *op) {
//                           return getUnscheduledUserCount(op, unScheduledOps)
//                           == 0;});
//   for (auto op = dominantOps.rbegin(); op != dominantOps.rend(); ++op) {
//     for (auto user : (*op)->getUsers()) {
//       if (isReadyToSchedule(user, unScheduledOps) &&
//           isLgSupport(user) == flag_local_turn) {
//         nextOp = user;
//         return true;
//       }
//     }
//   }
//   nextOp = nullptr;
//   return false;
// }

// static bool tryFirstUnscheduled(Operation *&nextOp, llvm::SetVector<Operation
// *> &unScheduledOps,
//                                  const bool flag_local_turn) {
//   for (auto op = unScheduledOps.rbegin(); op != unScheduledOps.rend(); ++op)
//   {
//     if (isReadyToSchedule(*op, unScheduledOps) &&
//         isLgSupport(*op) == flag_local_turn) {
//       nextOp = *op;
//       return true;
//     }
//   }
//   nextOp = nullptr;
//   return false;
// }

namespace SortStragegies {
/**
 * SORT STRANGIES:
 * 1. depth first PARENT-CHILD-PARENT-CHILD-...
 * 2. dominant node first for multi-branch structure
 * 3. Keep the original order and sort
 * NOTE: flag_local_turn for PING-PONG rule
 **/
static bool isLgSupport(Operation *op) {
  bool res = false;
  // if (isa<top::WeightOp>(op)) {
  //   res = true;
  // }
  if (auto lg_op = dyn_cast<tpu_mlir::LocalGenInterface>(op)) {
    res = mlir::succeeded(lg_op.LocalGenSupport());
  }
  return res;
}

static bool isReadyToSchedule(Operation *op,
                              llvm::SetVector<Operation *> &unScheduledOps) {
  if (!op || !unScheduledOps.contains(op))
    return false;
  for (auto in : op->getOperands()) {
    auto in_op = in.getDefiningOp();
    if (!in_op) {
      continue;
    }
    if (unScheduledOps.contains(in_op)) {
      return false;
    }
  }
  return true;
};

static int
getUnscheduledUserCount(Operation *op,
                        llvm::SetVector<Operation *> &unScheduledOps) {
  llvm::DenseSet<Operation *> unique_users;
  for (auto out : op->getResults()) {
    for (auto user : out.getUsers()) {
      if (unScheduledOps.contains(user)) {
        unique_users.insert(user);
      }
    }
  }
  return unique_users.size();
};

// A "heavy" op carries a large constant weight; 1MB is a tuning knob.
static constexpr double HEAVY_WEIGHT_MB = 1.0;

static bool isHeavyOp(Operation *op) {
  if (!isa<top::MatMulOp, top::ConvOp, top::DeconvOp, tpu::MatMulOp,
           tpu::Conv2DOp, tpu::DeconvOp>(op)) {
    return false;
  }
  for (auto in : op->getOperands()) {
    auto weight_op = dyn_cast_or_null<top::WeightOp>(in.getDefiningOp());
    if (!weight_op) {
      continue;
    }
    auto type =
        weight_op.getOutput().getType().dyn_cast<mlir::RankedTensorType>();
    if (!type) {
      continue;
    }
    double bytes =
        type.getNumElements() *
        module::getStorageType(weight_op.getOutput()).getIntOrFloatBitWidth() /
        8.0;
    if (bytes / 1024.0 / 1024.0 >= HEAVY_WEIGHT_MB) {
      return true;
    }
  }
  return false;
};

static bool heavyTopoSortDisabled() {
  static const bool disabled = getenv("TPU_DISABLE_HEAVY_TOPO_SORT") != nullptr;
  return disabled;
}

// Sibling branches are decided by reachability, not traversal order: ops
// reachable from nextOp form its downstream cone, and a heavy op inside the
// cone is reached naturally by depth-first, so only a heavy op outside the
// cone is a real sibling worth jumping to. Joins (ops with >= 2 real data
// producers, weight/none placeholders excluded) are collected on the way.
static void collectDownstreamCone(Operation *nextOp,
                                  llvm::DenseSet<Operation *> &cone,
                                  llvm::SmallVectorImpl<Operation *> &joins) {
  llvm::SmallVector<Operation *, 16> work{nextOp};
  cone.insert(nextOp);
  while (!work.empty()) {
    Operation *cur = work.pop_back_val();
    int producers = 0;
    for (auto in : cur->getOperands()) {
      auto prod = in.getDefiningOp();
      if (prod && !isa<top::WeightOp, top::NoneOp>(prod)) {
        producers++;
      }
    }
    if (producers >= 2) {
      joins.push_back(cur);
    }
    for (auto user : cur->getUsers()) {
      if (cone.insert(user).second) {
        work.push_back(user);
      }
    }
  }
}

static Operation *
findUnscheduledHeavy(Operation *root, const llvm::DenseSet<Operation *> &cone,
                     llvm::SetVector<Operation *> &unScheduledOps) {
  llvm::SmallDenseSet<Operation *> seen;
  llvm::SmallVector<Operation *, 8> stack{root};
  seen.insert(root);
  while (!stack.empty()) {
    Operation *p = stack.pop_back_val();
    if (cone.contains(p) || !unScheduledOps.contains(p)) {
      continue;
    }
    if (isHeavyOp(p)) {
      return p;
    }
    for (auto in : p->getOperands()) {
      auto up = in.getDefiningOp();
      if (up && seen.insert(up).second) {
        stack.push_back(up);
      }
    }
  }
  return nullptr;
}

// First schedulable op on the path to heavy. Scheduling it lets depth-first
// walk down the sibling branch so heavy ops cluster together. In a DAG such
// an op always exists, and it can never be inside nextOp's cone: a cone
// member upstream of heavy would make heavy reachable from nextOp.
static Operation *
findBranchFrontier(Operation *heavy,
                   llvm::SetVector<Operation *> &unScheduledOps) {
  llvm::SmallDenseSet<Operation *> seen;
  llvm::SmallVector<Operation *, 8> stack{heavy};
  seen.insert(heavy);
  while (!stack.empty()) {
    Operation *p = stack.pop_back_val();
    if (!unScheduledOps.contains(p)) {
      continue;
    }
    if (isReadyToSchedule(p, unScheduledOps)) {
      return p;
    }
    for (auto in : p->getOperands()) {
      auto up = in.getDefiningOp();
      if (up && seen.insert(up).second) {
        stack.push_back(up);
      }
    }
  }
  return nullptr;
}

// Returns the frontier op of an independent sibling branch that still has an
// unscheduled heavy op, or nullptr when there is no such branch.
static Operation *
findHeavySiblingFrontier(Operation *nextOp,
                         llvm::SetVector<Operation *> &unScheduledOps) {
  llvm::DenseSet<Operation *> cone;
  llvm::SmallVector<Operation *, 8> joins;
  collectDownstreamCone(nextOp, cone, joins);
  for (auto join : joins) {
    for (auto in : join->getOperands()) {
      auto prod = in.getDefiningOp();
      if (!prod || isa<top::WeightOp, top::NoneOp>(prod) ||
          cone.contains(prod)) {
        continue;
      }
      if (auto heavy = findUnscheduledHeavy(prod, cone, unScheduledOps)) {
        if (auto frontier = findBranchFrontier(heavy, unScheduledOps)) {
          return frontier;
        }
      }
    }
  }
  return nullptr;
}

class BaseStragegy {
public:
  virtual ~BaseStragegy() = default;
  virtual bool applySort(Operation *&nextOp, Operation *prevOp,
                         llvm::SetVector<Operation *> &unScheduledOps,
                         llvm::SetVector<Operation *> &dominantOps,
                         const bool flag_sort_local_ops) = 0;
};

class depthFirstStrategy : public BaseStragegy {
public:
  bool applySort(Operation *&nextOp, Operation *prevOp,
                 llvm::SetVector<Operation *> &unScheduledOps,
                 llvm::SetVector<Operation *> &dominantOps,
                 const bool flag_sort_local_ops) override {
    if (!prevOp) {
      // A function always starts with prevOp == nullptr; snapshot its heavy
      // ops here so later calls can skip scouting once all are scheduled.
      heavyOps.clear();
      for (auto op : unScheduledOps) {
        if (isHeavyOp(op)) {
          heavyOps.push_back(op);
        }
      }
      return false;
    }
    if (prevOp->hasOneUse() == true) {
      nextOp = *(prevOp->user_begin());
      if (isReadyToSchedule(nextOp, unScheduledOps) &&
          isLgSupport(nextOp) == flag_sort_local_ops) {
        if (!heavyTopoSortDisabled() && hasLiveHeavy(unScheduledOps)) {
          if (auto frontier =
                  findHeavySiblingFrontier(nextOp, unScheduledOps)) {
            if (isLgSupport(frontier) == flag_sort_local_ops) {
              nextOp = frontier;
              return true;
            }
            // The frontier belongs to the other LG class: skip nextOp this
            // round so the ping-pong flag can flip and make the frontier
            // schedulable. This is best-effort only; the ping-pong fallback
            // may still pick the held-back op first.
            nextOp = nullptr;
            return false;
          }
        }
        return true;
      }
    }
    nextOp = nullptr;
    return false;
  }

private:
  bool hasLiveHeavy(llvm::SetVector<Operation *> &unScheduledOps) {
    for (auto h : heavyOps) {
      if (unScheduledOps.contains(h)) {
        return true;
      }
    }
    return false;
  }

  llvm::SmallVector<Operation *, 8> heavyOps;
};

class dominantNodeFirstStrategy : public BaseStragegy {
public:
  bool applySort(Operation *&nextOp, Operation *prevOp,
                 llvm::SetVector<Operation *> &unScheduledOps,
                 llvm::SetVector<Operation *> &dominantOps,
                 const bool flag_sort_local_ops) override {
    dominantOps.remove_if([&](Operation *op) {
      return getUnscheduledUserCount(op, unScheduledOps) == 0;
    });
    for (auto op = dominantOps.rbegin(); op != dominantOps.rend(); ++op) {
      for (auto user : (*op)->getUsers()) {
        if (isReadyToSchedule(user, unScheduledOps) &&
            isLgSupport(user) == flag_sort_local_ops) {
          nextOp = user;
          return true;
        }
      }
    }
    nextOp = nullptr;
    return false;
  }
};

class simplePingPongStrategy : public BaseStragegy {
public:
  bool applySort(Operation *&nextOp, Operation *prevOp,
                 llvm::SetVector<Operation *> &unScheduledOps,
                 llvm::SetVector<Operation *> &dominantOps,
                 const bool flag_sort_local_ops) override {
    for (auto op = unScheduledOps.rbegin(); op != unScheduledOps.rend(); ++op) {
      if (isReadyToSchedule(*op, unScheduledOps) &&
          isLgSupport(*op) == flag_sort_local_ops) {
        nextOp = *op;
        return true;
      }
    }
    nextOp = nullptr;
    return false;
  }
};
} // namespace SortStragegies
using namespace SortStragegies;

class TopoSortPass : public TopoSortBase<TopoSortPass> {
public:
  TopoSortPass() {}
  void runOnOperation() override {
    if (module::isPlatform(module::Platform::LLM) ||
        module::isPlatform(module::Platform::LLM_QUANTIZED)) {
      return;
    }
    // init
    std::vector<std::unique_ptr<BaseStragegy>> strategies;
    if (depth_first == "true") {
      strategies.push_back(std::make_unique<depthFirstStrategy>());
      strategies.push_back(std::make_unique<dominantNodeFirstStrategy>());
    }
    strategies.push_back(std::make_unique<simplePingPongStrategy>());
    // gogo
    auto modules = module::getAllModules();
    for (auto s : *modules) {
      for (auto func : s.getOps<FuncOp>()) {
        if (func.getName() == "main") {
          continue;
        }
        // assume one block only.
        Block &entry_block = func.getBody().front();
        sortTopologically(
            strategies, &entry_block,
            llvm::make_range(entry_block.begin(), entry_block.end()));
      }
    }
  }

private:
  // PING-PONG: collect glb ops -> local ops -> glb ops -> local ops -> ...
  void sortTopologically(std::vector<std::unique_ptr<BaseStragegy>> &strategies,
                         Block *block,
                         llvm::iterator_range<Block::iterator> ops) const {

    bool flag_pick_local_ops = true;
    llvm::SetVector<Operation *> unScheduledOps;
    llvm::SetVector<Operation *> dominantOps; // dominant ops (in order)
    for (auto &op : ops) {
      if (!isa<top::WeightOp, top::NoneOp, func::ReturnOp>(&op)) {
        unScheduledOps.insert(&op);
      }
    }

    Operation *prevOp = nullptr, *nextOp = nullptr;
    // int ii = 0;
    while (!unScheduledOps.empty()) {
      // try sort strategies
      bool success = false;
      for (auto &strategy : strategies) {
        success = strategy->applySort(nextOp, prevOp, unScheduledOps,
                                      dominantOps, flag_pick_local_ops);
        if (success) {
          break;
        }
      }
      if (success) {
        assert(unScheduledOps.contains(nextOp));
        unScheduledOps.remove(nextOp);
        if (prevOp) {
          nextOp->moveAfter(prevOp);
        }
        prevOp = nextOp;
        if (nextOp->hasOneUse() == false &&
            getUnscheduledUserCount(nextOp, unScheduledOps) > 0) {
          dominantOps.insert(nextOp);
        }
      } else {
        flag_pick_local_ops = !flag_pick_local_ops;
      }
    }

    // Schedule WeightOp, NoneOp
    for (auto &op : ops) {
      if (isa<top::WeightOp>(&op)) {
        Operation *firstUser = nullptr;
        for (auto user : op.getUsers()) {
          if (!firstUser || user->isBeforeInBlock(firstUser)) {
            firstUser = user;
          }
        }
        if (firstUser) {
          op.moveBefore(firstUser);
        }
      } else if (isa<top::NoneOp>(&op)) {
        op.moveBefore(&(*block->begin()));
      } else if (isa<func::ReturnOp>(&op)) {
        op.moveAfter(&(*std::prev(block->end())));
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createTopoSortPass() {
  return std::make_unique<TopoSortPass>();
}
} // namespace tpu
} // namespace tpu_mlir

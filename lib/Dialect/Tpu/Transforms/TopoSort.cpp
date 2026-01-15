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
    if (!prevOp)
      return false;
    if (prevOp->hasOneUse() == true) {
      nextOp = *(prevOp->user_begin());
      if (isReadyToSchedule(nextOp, unScheduledOps) &&
          isLgSupport(nextOp) == flag_sort_local_ops) {
        return true;
      }
    }
    nextOp = nullptr;
    return false;
  }
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

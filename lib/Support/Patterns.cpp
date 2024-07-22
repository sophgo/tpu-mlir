//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/Patterns.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

namespace tpu_mlir {
namespace patterns {

// if 2 op is same, fuse it.
LogicalResult FuseSameOp::matchAndRewriteImpl(Operation *op,
                                          PatternRewriter &rewriter) const {
  if (isa_and_nonnull<top::NoneOp, top::WeightOp>(op)) {
    return failure();
  }
  auto users = op->getUsers();
  auto num_users = std::distance(users.begin(), users.end());
  if (num_users < 2) {
    return failure();
  }
  for (auto first = op->user_begin(); first != op->user_end(); first++) {
    auto it = first;
    for (it++; it != users.end(); it++) {
      if (*first == *it) {
        continue;
      }
      if (module::isSameOp(*first, *it)) {
        if ((*first)->isBeforeInBlock(*it)) {
          (*it)->replaceAllUsesWith(*first);
          rewriter.eraseOp(*it);
        } else {
          (*first)->replaceAllUsesWith(*it);
          rewriter.eraseOp(*first);
        }
        return success();
      }
    }
  }
  return failure();
}
} // namespace patterns
} // namespace tpu_mlir

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/Patterns.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"

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
        NameLoc loc_name;
        if ((*first)->isBeforeInBlock(*it)) {
          bool is_rename = module::isOpBlockReturnOp(*it);
          if (is_rename) {
            loc_name = module::getLoc(it->getResult(0));
          }
          (*it)->replaceAllUsesWith(*first);
          rewriter.eraseOp(*it);
          if (is_rename) {
            module::setLoc(first->getResult(0), loc_name);
          }
        } else {
          bool is_rename = module::isOpBlockReturnOp(*first);
          if (is_rename) {
            loc_name = module::getLoc(first->getResult(0));
          }
          (*first)->replaceAllUsesWith(*it);
          rewriter.eraseOp(*first);
          if (is_rename) {
            module::setLoc(it->getResult(0), loc_name);
          }
        }
      }
      return success();
    }
  }
  // }
  return failure();
}

// for input of batchnormtrain and batchnormbwd
// insert reshape (cxdtype) -> (1xcx1x1xdtype)
LogicalResult
InputReshape::matchAndRewriteImpl(Operation *op,
                                  PatternRewriter &rewriter) const {
  if (!isa<top::BatchNormBwdOp, top::BatchNormTrainOp>(op)) {
    return failure();
  }
  bool reshape_insert = false;
  for (int i = 0; i < op->getOperands().size(); i++) {
    auto in = op->getOperand(i);
    auto in_shape = module::getShape(in);
    if (in_shape.size() == 1) {
      auto reshape_v = rewriter.create<top::ReshapeOp>(
          NameLoc::get(
              rewriter.getStringAttr(module::getName(in).str() + "_reshape")),
          RankedTensorType::get({1, in_shape[0], 1, 1},
                                module::getElementType(in)),
          in);
      op->setOperand(i, reshape_v);
      reshape_insert = true;
    }
  }

  return reshape_insert ? success() : failure();
}

} // namespace patterns
} // namespace tpu_mlir

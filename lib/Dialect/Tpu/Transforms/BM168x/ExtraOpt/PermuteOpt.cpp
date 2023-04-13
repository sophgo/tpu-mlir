//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DoExtraOpt.h"
#include "tpu_mlir/Support/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include <fstream>
#include <set>
#include <sstream>

using namespace llvm;

namespace tpu_mlir {
namespace bm1684x {

// reorder op when transpose is before mulconst/cast/softmax to optimize bert
LogicalResult
PermuteReorderPattern::matchAndRewrite(tpu::PermuteOp op,
                                       PatternRewriter &rewriter) const {
//  if (module::isAsymmetric()) {
//    return failure();
//  }

  if (op->hasOneUse() == false) {
    return failure();
  }
  std::vector<int64_t> ps = {0, 2, 1, 3};
  auto order = module::getI64Array(op.getOrder());
  if (*order != ps) {
    return failure();
  }

  auto in_shape = module::getShape(op.getInput());
  auto out_shape = module::getShape(op.getOutput());
  auto nextOp = *op.getOutput().getUsers().begin();
  if (nextOp->hasOneUse() == false) {
    return failure();
  }
  if (auto mulconst_op = dyn_cast<tpu::MulConstOp>(nextOp)) {
    auto newType = RankedTensorType::get(
        in_shape, module::getElementType(mulconst_op.getOutput()));
    mulconst_op.getOutput().setType(newType);
    op.replaceAllUsesWith(op.getInput());
    rewriter.setInsertionPointAfter(mulconst_op);
    newType = RankedTensorType::get(
        out_shape, module::getElementType(mulconst_op.getOutput()));
    auto out_loc = mulconst_op.getLoc(); // keep out location unchanged.
    auto name = module::getName(mulconst_op.getOutput());
    auto loc = NameLoc::get(rewriter.getStringAttr(name + "_trans"));
    mulconst_op->setLoc(loc);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(ps)));
    auto new_op = rewriter.create<tpu::PermuteOp>(
        out_loc, newType,
        ValueRange{mulconst_op.getOutput(), module::getNoneOp(mulconst_op)},
        attrs);
    mulconst_op.getOutput().replaceAllUsesExcept(new_op.getOutput(), {new_op});
    return success();
  } else if (auto cast_op = dyn_cast<tpu::CastOp>(nextOp)) {
    auto newType = RankedTensorType::get(
        in_shape, module::getElementType(cast_op.getOutput()));
    cast_op.getOutput().setType(newType);
    op.replaceAllUsesWith(op.getInput());
    rewriter.setInsertionPointAfter(cast_op);
    newType = RankedTensorType::get(
        out_shape, module::getElementType(cast_op.getOutput()));
    auto out_loc = cast_op.getLoc(); // keep out location unchanged.
    auto name = module::getName(cast_op.getOutput());
    auto loc = NameLoc::get(rewriter.getStringAttr(name + "_trans"));
    cast_op->setLoc(loc);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(ps)));
    auto new_op = rewriter.create<tpu::PermuteOp>(
        out_loc, newType,
        ValueRange{cast_op.getOutput(), module::getNoneOp(cast_op)}, attrs);
    cast_op.getOutput().replaceAllUsesExcept(new_op.getOutput(), {new_op});
    return success();
  } else if (auto softmax_op = dyn_cast<tpu::SoftmaxOp>(nextOp)) {
    int64_t axis = softmax_op.getAxis();
    if (!(axis == -1 || axis == out_shape.size() - 1)) {
      return failure();
    }
    auto newType = RankedTensorType::get(
        in_shape, module::getElementType(softmax_op.getOutput()));
    softmax_op.getOutput().setType(newType);
    op.replaceAllUsesWith(op.getInput());
    rewriter.setInsertionPointAfter(softmax_op);
    newType = RankedTensorType::get(
        out_shape, module::getElementType(softmax_op.getOutput()));
    auto out_loc = cast_op.getLoc(); // keep out location unchanged.
    auto name = module::getName(softmax_op.getOutput());
    auto loc = NameLoc::get(rewriter.getStringAttr(name + "_trans"));
    softmax_op->setLoc(loc);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(ps)));
    auto new_op = rewriter.create<tpu::PermuteOp>(
        out_loc, newType,
        ValueRange{softmax_op.getOutput(), module::getNoneOp(softmax_op)},
        attrs);
    softmax_op.getOutput().replaceAllUsesExcept(new_op.getOutput(), {new_op});
    return success();
  } else if (auto permute_op = dyn_cast<tpu::PermuteOp>(nextOp)) {
    auto next_order = module::getI64Array(op.getOrder());
    if (*next_order != ps) {
      return failure();
    }
    permute_op.replaceAllUsesWith(op.getInput());
    // op.replaceAllUsesWith(op.geInput());
    rewriter.eraseOp(permute_op);
    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}

} // namespace bm1684x
} // namespace tpu_mlir

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/ExtraConversion/ExtraConvertCV18XX.h"
#include "tpu_mlir/Support/MathUtils.h"

namespace tpu_mlir {
namespace cv18xx {

LogicalResult ConvertMatMulWithRightTranspose::matchAndRewrite(
    top::MatMulOp op, PatternRewriter &rewriter) const {
  auto left = op.getInput();
  auto right = op.getRight();
  auto out = module::getNextOp(op);

  std::vector<int64_t> lshape = module::getShape(left);
  std::vector<int64_t> rshape = module::getShape(right);
  if (lshape.size() != 4 || rshape.size() != 4) {
    return failure();
  }
  if (op.getRightTranspose() != false || op.getLeftTranspose() != false ||
      op.getOutputTranspose() != false) {
    return failure();
  }

  bool match = false;
  std::vector<int64_t> pattern = {0, 2, 1, 3};
  std::vector<int64_t> shape_4, order_4;
  auto leftOp = left.getDefiningOp();
  if (isa<top::PermuteOp>(leftOp) && left.hasOneUse()) {
    auto ltrans_op = dyn_cast<top::PermuteOp>(leftOp);
    order_4 = *module::getI64Array(ltrans_op.getOrder());
    if (order_4 == pattern) {
      op.setOperand(0, ltrans_op.getInput());
      op.setLeftTranspose(true);
      op.setHdimIsBatch(true);
      rewriter.eraseOp(ltrans_op);
      match = true;
    }
  }
  auto rightOp = right.getDefiningOp();
  if (isa<top::PermuteOp>(rightOp) && right.hasOneUse()) {
    auto rtrans_op = dyn_cast<top::PermuteOp>(rightOp);
    order_4 = *module::getI64Array(rtrans_op.getOrder());
    if (order_4 == pattern) {
      op.setOperand(1, rtrans_op.getInput());
      op.setRightTranspose(true);
      op.setHdimIsBatch(true);
      rewriter.eraseOp(rtrans_op);
      match = true;
    }
  }
  if (out != nullptr && isa<top::PermuteOp>(out) &&
      op.getResult().hasOneUse()) {
    auto otrans_op = dyn_cast<top::PermuteOp>(out);
    order_4 = *module::getI64Array(otrans_op.getOrder());
    if (order_4 == pattern) {
      op.setOutputTranspose(true);
      op.setHdimIsBatch(true);
      op->setLoc(otrans_op->getLoc());
      op.getResult().setType(otrans_op.getResult().getType());
      otrans_op.getOutput().replaceAllUsesWith(otrans_op.getInput());
      rewriter.eraseOp(otrans_op);
      match = true;
    }
  }
  return match ? success() : failure();
}

} // namespace cv18xx
} // namespace tpu_mlir

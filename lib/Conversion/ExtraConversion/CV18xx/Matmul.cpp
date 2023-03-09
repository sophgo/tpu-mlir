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
  auto filter = op.getRight();
  if (module::isWeight(filter)) {
    return failure();
  }
  if (filter.hasOneUse() == false) {
    return failure();
  }
  auto trans_op = dyn_cast<top::PermuteOp>(filter.getDefiningOp());
  if (!trans_op) {
    return failure();
  }
  if (op.getRightTranspose()) {
    return failure();
  }

  bool match = false;
  std::vector<int64_t> pattern = {0, 2, 1, 3};
  std::vector<int64_t> order_4;
  std::vector<int64_t> shape_4;

  std::vector<int64_t> shape = module::getShape(trans_op.getInput());
  auto order = module::getI64Array(trans_op.getOrder());
  auto ret = permute_reset(shape, *order, shape_4, order_4, 4);
  if (ret == false) {
    return failure();
  }
  if (order_4 == pattern) {
    op.setOperand(1, trans_op.getInput());
    op.setRightTranspose(true);
    rewriter.eraseOp(trans_op);
    match = true;
  }
  return match ? success() : failure();
}

} // namespace cv18xx
} // namespace tpu_mlir

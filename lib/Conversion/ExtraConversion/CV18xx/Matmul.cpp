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
  std::vector<int64_t> lshape = module::getShape(left);
  std::vector<int64_t> rshape = module::getShape(right);
  auto out = module::getNextOp(op);

  //if right is weight and need transpose, do it here.
  if (op.getRightTranspose() && isa<top::WeightOp>(right.getDefiningOp())) {
    std::string filter_name = module::getName(right).str();
    auto filterOp = dyn_cast<top::WeightOp>(right.getDefiningOp());
    auto filter_f32 = filterOp.read<float>();
    int64_t filter_dims = rshape.size();
    auto p = op.parseParam();
    int64_t K = p.K, N = p.N;
    int64_t batch = std::accumulate(rshape.begin(), rshape.end() - 2,
                                    1, std::multiplies<int64_t>());
    assert(rshape[filter_dims - 2] == N && rshape[filter_dims - 1] == K && batch == p.batch);
    //transpose filter of last two dims
    std::vector<float> new_filter(filter_f32->size());
    for (int64_t b = 0; b < batch; b++) {
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
          int64_t idx1 = b * N * K + i * K + j;
          int64_t idx2 = b * N * K + j * N + i;
          new_filter[idx1] = filter_f32->at(idx2);
        }
      }
    }
    //swap last two dims to get new_rshape
    std::swap(rshape[filter_dims - 2], rshape[filter_dims - 1]);
    auto new_filter_type = RankedTensorType::get(rshape, rewriter.getF32Type());
    auto new_filter_op = top::WeightOp::create(op, filter_name, new_filter, new_filter_type);
    op.setOperand(1, new_filter_op);
    op.setRightTranspose(false);
    return success();
  }

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

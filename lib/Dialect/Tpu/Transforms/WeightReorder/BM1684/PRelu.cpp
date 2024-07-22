//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../WeightReorder.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"

using namespace bm1684;
using namespace backend;

template <>
LogicalResult WeightReorder<tpu::PReluOp, int8_t>::matchAndRewriteImpl(
    tpu::PReluOp op, PatternRewriter &rewriter) const {
  // convert 1N to 4N
  if (module::isWeight(op.getSlope()) &&
      module::getStorageType(op.getSlope()).isInteger(8) &&
      module::getNumElements(op.getSlope()) > 1) {
    auto slopeOp = cast<top::WeightOp>(op.getSlope().getDefiningOp());
    auto value_ptr = slopeOp.read<int8_t>();
    auto value_new = BM1684::instance().Convert1NTo4N(op.getSlope(), value_ptr);
    auto tensor_type = op.getSlope().getType().cast<RankedTensorType>();
    auto new_slope = top::WeightOp::create(slopeOp, "reorderd", *value_new,
                                           tensor_type, STORE_MODE_4N);
    op.setOperand(1, new_slope);
    return success();
  }
  return failure();
}

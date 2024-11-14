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
LogicalResult WeightReorder<tpu::ScaleOp, int8_t>::matchAndRewriteImpl(
    tpu::ScaleOp op, PatternRewriter &rewriter) const {
  // convert 1N to 4N
  if (module::isWeight(op.getScale()) &&
      module::getStorageType(op.getScale()).isInteger(8) &&
      module::isWeight(op.getBias()) &&
      module::getStorageType(op.getBias()).isInteger(16) &&
      module::isWeight(op.getLshift()) &&
      module::getStorageType(op.getLshift()).isInteger(8)) {
    auto scaleOp = cast<top::WeightOp>(op.getScale().getDefiningOp());
    auto scale_ptr = scaleOp.read<int8_t>();
    auto scale_new = BM1684::instance().Convert1NTo4N(op.getScale(), scale_ptr);
    auto scale_type = op.getScale().getType().cast<RankedTensorType>();
    auto new_scale = top::WeightOp::create(scaleOp, "reorderd", *scale_new,
                                           scale_type, STORE_MODE_4N);
    op.setOperand(1, new_scale);

    // convert 1N to 2N
    auto biasOp = cast<top::WeightOp>(op.getBias().getDefiningOp());
    auto bias_ptr = biasOp.read<int16_t>();
    auto bias_new = BM1684::instance().Convert1NTo2N(op.getBias(), bias_ptr);
    auto bias_type = op.getBias().getType().cast<RankedTensorType>();
    auto new_bias = top::WeightOp::create(biasOp, "reorderd", *bias_new,
                                          bias_type, STORE_MODE_2N);
    op.setOperand(2, new_bias);

    // convert 1N to 4N
    auto lshiftOp = cast<top::WeightOp>(op.getLshift().getDefiningOp());
    auto lshift_ptr = lshiftOp.read<int8_t>();
    auto lshift_new =
        BM1684::instance().Convert1NTo4N(op.getLshift(), lshift_ptr);
    auto lshift_type = op.getLshift().getType().cast<RankedTensorType>();
    auto new_lshift = top::WeightOp::create(lshiftOp, "reorderd", *lshift_new,
                                            lshift_type, STORE_MODE_4N);
    op.setOperand(3, new_lshift);
    return success();
  }
  return failure();
}

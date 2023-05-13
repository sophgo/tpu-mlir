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
LogicalResult WeightReorder<tpu::MulOp, int8_t>::matchAndRewrite(
    tpu::MulOp op, PatternRewriter &rewriter) const {
  /// convert 1N to 4N
  for (int32_t i = 0; i < op.getNumOperands(); i++) {
    if (module::isWeight(op.getInputs()[i]) &&
        module::getStorageType(op.getInputs()[i]).isInteger(8)) {
      auto castOp = cast<top::WeightOp>(op.getInputs()[i].getDefiningOp());
      auto value = op.getInputs()[i];
      auto value_ptr = castOp.read<int8_t>();
      auto value_new = BM1684::instance().Convert1NTo4N(value, value_ptr);
      auto tensor_type = value.getType().cast<RankedTensorType>();
      auto newOp = top::WeightOp::create(castOp, "reorderd", *value_new,
                                         tensor_type, STORE_MODE_4N);
      op.setOperand(i, newOp);
      return success();
    }
  }
  return failure();
}

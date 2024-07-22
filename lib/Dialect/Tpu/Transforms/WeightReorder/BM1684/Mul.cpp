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
LogicalResult WeightReorder<tpu::MulOp, int8_t>::matchAndRewriteImpl(
    tpu::MulOp op, PatternRewriter &rewriter) const {
  /// convert 1N to 4N
  for (int32_t i = 0; i < op.getNumOperands(); i++) {
    if (module::isWeight(op.getInputs()[i]) &&
        module::getStorageType(op.getInputs()[i]).isInteger(8)) {
      auto castOp = cast<top::WeightOp>(op.getInputs()[i].getDefiningOp());
      rewriter.setInsertionPointAfter(castOp);
      auto name = module::getName(op.getInputs()[i]);
      auto new_loc = NameLoc::get(
          rewriter.getStringAttr(name.str() + "_convert_to_activation"));

      auto value = op.getInputs()[i];
      auto value_ptr = castOp.read<int8_t>();
      auto value_new = BM1684::instance().Convert1NTo4N(value, value_ptr);
      auto tensor_type = value.getType().cast<RankedTensorType>();
      auto new_weight_value = top::WeightOp::create(
          castOp, "reorderd", *value_new, tensor_type, STORE_MODE_4N);
      auto weight2activation_op = rewriter.create<tpu::Weight2ActivationOp>(
          castOp.getLoc(), tensor_type, ValueRange{new_weight_value});

      castOp->setLoc(new_loc);
      castOp.replaceAllUsesWith(weight2activation_op.getOperation());
      return success();
    }
  }
  return failure();
}

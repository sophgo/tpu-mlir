//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct TopGroupNormReshape : public OpRewritePattern<GroupNormOp> {
  using OpRewritePattern::OpRewritePattern;
  TopGroupNormReshape(MLIRContext *context)
      : OpRewritePattern<GroupNormOp>(context) {}

  LogicalResult matchAndRewrite(GroupNormOp op,
                                PatternRewriter &rewriter) const override {
    auto in_shape = module::getShape(op.getInput());
    auto num_dims = in_shape.size();
    LogicalResult res = failure();

    auto weight = op.getWeight();
    if (module::isWeight(weight)) {
      auto weight_shape = module::getShape(weight);
      if (num_dims != weight_shape.size()) {
        std::vector<int64_t> new_shape(num_dims, 1);
        new_shape[1] = weight_shape[0];
        auto new_type =
            RankedTensorType::get(new_shape, module::getElementType(weight));
        weight.setType(new_type);
        res = success();
      }
    }

    auto bias = op.getBias();
    if (module::isWeight(bias)) {
      auto bias_shape = module::getShape(bias);
      if (num_dims != bias_shape.size()) {
        std::vector<int64_t> new_shape(num_dims, 1);
        new_shape[1] = bias_shape[0];
        auto new_type =
            RankedTensorType::get(new_shape, module::getElementType(bias));
        bias.setType(new_type);
        res = success();
      }
    }

    return res;
  }
};

void GroupNormOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<TopGroupNormReshape>(context);
}

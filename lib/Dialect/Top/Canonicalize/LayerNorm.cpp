//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct TopLayerNormReshape : public OpRewriterPatternEx<LayerNormOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  TopLayerNormReshape(mlir::MLIRContext *context)
      : OpRewriterPatternEx<LayerNormOp>(context, "TopLayerNormReshape") {}

  LogicalResult matchAndRewriteImpl(LayerNormOp op,
                                    PatternRewriter &rewriter) const override {
    auto in_shape = module::getShape(op.getInput());
    auto num_dims = in_shape.size();
    auto axis = op.getAxis();
    LogicalResult res = failure();

    auto weight = op.getWeight();
    if (module::isWeight(weight)) {
      auto weight_shape = module::getShape(weight);
      if (num_dims != weight_shape.size()) {
        std::vector<int64_t> new_shape(num_dims, 1);
        for (int64_t i = axis; i < num_dims; ++i) {
          new_shape[i] = weight_shape[i - axis];
        }
        module::setShape(weight, new_shape);
        res = success();
      }
    }

    auto bias = op.getBias();
    if (module::isWeight(bias)) {
      auto bias_shape = module::getShape(bias);
      if (num_dims != bias_shape.size()) {
        std::vector<int64_t> new_shape(num_dims, 1);
        for (int64_t i = axis; i < num_dims; ++i) {
          new_shape[i] = bias_shape[i - axis];
        }
        module::setShape(bias, new_shape);
        res = success();
      }
    }

    return res;
  }
};

void LayerNormOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<TopLayerNormReshape>(context);
}

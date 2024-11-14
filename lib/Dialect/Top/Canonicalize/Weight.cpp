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

// all data is same
struct WeightToConst : public OpRewriterPatternEx<top::WeightOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  WeightToConst(mlir::MLIRContext *context)
      : OpRewriterPatternEx<top::WeightOp>(context, "WeightToConst") {}

  LogicalResult matchAndRewriteImpl(top::WeightOp op,
                                    PatternRewriter &rewriter) const override {
    if (!op->hasOneUse()) {
      return failure();
    }
    auto stype = module::getStorageType(op);
    if (!stype.isF32()) {
      return failure();
    }
    auto user = *op->user_begin();
    if (!user->hasTrait<trait::SupportConstant>()) {
      return failure();
    }
    // avoid broadcast
    auto out = user->getOpResult(0);
    auto out_shape = module::getShape(out);
    for (auto in : user->getOperands()) {
      if (module::isWeight(in)) {
        continue;
      }
      auto in_shape = module::getShape(in);
      if (in_shape != out_shape) {
        return failure();
      }
    }
    auto data = op.read<float>();
    if (data->size() <= 1) {
      return failure();
    }
    auto d0 = data->at(0);
    for (auto &d : *data) {
      if (d != d0) {
        return failure();
      }
    }
    std::vector<float> new_data = {d0};
    auto new_type = module::getTypeLike(op.getOutput(), {1});
    auto weight = top::WeightOp::create(user, "_constant", new_data, new_type);
    rewriter.replaceOp(op, {weight});
    return success();
  }
};

void top::WeightOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<WeightToConst>(context);
}

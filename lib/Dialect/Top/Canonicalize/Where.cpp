//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace tpu_mlir::top;

// in gpt2 model, the mask to softmax is from where, very small value in weight tensor, change them to -10000
struct FilterWhereWeightPattern : public OpRewritePattern<WhereOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WhereOp op,
                                PatternRewriter &rewriter) const override {
    if (module::isUniformQuantized(op.getOutput()))
      return failure();
    if (!op->hasOneUse()) {
      return failure();
    }
    int in_cnt = 0;
    int weight_cnt = 0;
    WeightOp weight_op[2] = {NULL};
    SoftmaxOp softmax_op = NULL;
    for (auto opd:op.getOperands()){
      if (weight_op[weight_cnt] = dyn_cast<WeightOp>(opd.getDefiningOp())) {
        weight_cnt ++;
        if (weight_cnt > 2)
          return failure();
      }
      in_cnt ++;
    }
    if (in_cnt != 3 || weight_cnt != 2 || weight_op[0] == NULL || weight_op[1] == NULL)
      return failure();

    for (auto out:op.getOutput().getUsers())
      if (softmax_op = dyn_cast<SoftmaxOp>(out))
        break;
    if (softmax_op == NULL)
      return failure();

    for (int i=0;i<2;i++) {
      auto w = weight_op[i].read<float>();
      for (int i=0;i<w.get()->size();i++){
        if (w->at(i) < -3e38)
          w->at(i) = -10000;
      }
      const std::vector<float> tmp(*w);
      weight_op[i].update(tmp, (size_t)(w.get()->size()));
    }
    return success();
  }
};

void WhereOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<FilterWhereWeightPattern>(context);
}

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"


using namespace tpu_mlir::top;

struct TopFuseTile : public OpRewritePattern<TileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileOp op,
                                PatternRewriter &rewriter) const override {

    auto next_op = *op->user_begin();
    if (isa<AddOp, SubOp, MulOp, MinOp, MaxOp>(next_op)) {
      auto shape0 = module::getShape(op.getInput());
      auto shape1 = module::getShape(op.getOutput());
      for (int i = 0; i < shape0.size(); ++i) {
        if (shape0[i] != shape1[i] && std::min(shape0[i], shape1[i]) != 1)
          return failure();
      }
      // remove the Tile Op
      rewriter.replaceOp(op, {op.getInput()});
      return success();
    }
    return failure();
  }
};

struct ReplaceWithWeightInput : public OpRewritePattern<TileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileOp op,
                                PatternRewriter &rewriter) const override {

    if (isa<WeightOp>(op.getInput().getDefiningOp())) {
      auto weight = dyn_cast<WeightOp>(op.getInput().getDefiningOp());
      auto w = weight.read<float>();
      auto shape0 = module::getShape(op.getInput());
      auto shape1 = module::getShape(op.getOutput());
      bool updated = false;
      for (int i=shape0.size()-1;i>=0;i--) {
        if (shape0[i] == shape1[i])
          continue;
        int tile=shape1[i]/shape0[i];
        size_t inner = shape0[i];
        for (int j=i+1;j<shape1.size();j++) inner *= shape1[j];
        size_t outer = 1;
        for (int j=i-1;j>=0;j--) outer *= shape0[j];

        std::shared_ptr<std::vector<float>> new_w = std::make_shared<std::vector<float>>();
        for (int j=0;j<outer;j++) {
          for (int k=0;k<tile;k++) {
            new_w.get()->insert(new_w.get()->end(), w.get()->begin()+j*inner,w.get()->begin()+(j+1)*inner);
          }
        }
        w = new_w;
        updated = true;
      }
      if (updated) {
        auto type = RankedTensorType::get(shape1, rewriter.getF32Type());
        auto w_op = WeightOp::create<float>(op, module::getName(op.getOutput()).str(),*w, type);
        rewriter.replaceOp(op, w_op);
        return success();
      }
    }
    return failure();
  }
};

void TileOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<TopFuseTile, ReplaceWithWeightInput>(context);
}

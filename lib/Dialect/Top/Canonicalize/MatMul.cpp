//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace tpu_mlir::top;
using namespace tpu_mlir::helper;

// [4, 512, 1, 1] * [512, 1000] => [4, 512] * [512, 1000]
// test by caffe resnet18
struct MatMulReshape : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto in0_op = op.input().getDefiningOp();
    if (!in0_op->hasOneUse()) {
      return failure();
    }
    auto in1_op = op.right().getDefiningOp();
    if (false == isa<WeightOp>(in1_op)) {
      return failure();
    }
    auto shape0 = Module::getShape(op.input());
    auto shape1 = Module::getShape(op.right());
    if (shape0.size() == 4 && shape0[2] == 1 && shape0[3] == 1 && shape1.size() == 2) {
      auto etype = op.input().getType().cast<RankedTensorType>().getElementType();
      auto new_type = RankedTensorType::get({shape0[0], shape0[1]}, etype);
      op.input().setType(new_type);
      return success();
    }
    return failure();
  }
};

void MatMulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<MatMulReshape>(context);
}

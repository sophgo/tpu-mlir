//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//


#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::LoopOp::init(InferenceParameter &p) {
  return success();
}

void tpu::LoopOp::deinit(InferenceParameter &p) {
}

LogicalResult tpu::LoopOp::inference(InferenceParameter &p) {
  return success();
}

Operation::result_range tpu::LoopOp::v_final() {
  auto results = getResults();
  return llvm::make_range(
      results.begin(), results.begin() + getVInitial().size());
}

Operation::result_range tpu::LoopOp::scan_outputs() {
  auto results = getResults();
  return llvm::make_range(
      results.begin() + getVInitial().size(), results.end());
}

struct UpdateArgument : public OpRewritePattern<tpu::LoopOp> {
  UpdateArgument(mlir::MLIRContext *context)
      : OpRewritePattern<tpu::LoopOp>(context, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(tpu::LoopOp op,
                  mlir::PatternRewriter &rewriter) const override {
    for (int i = 0; i < op.getBody().getNumArguments(); i++) {
      auto type = op.getOperand(i).getType();
      op.getBody().getArgument(i).setType(type);
    }

    auto yieldOp = op.getBody().front().getTerminator();
    OpBuilder builder(module::getCtx());
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPoint(yieldOp);

    for (int i = 0; i < op.v_final().size() + 1; i++) {
      //check if need to insert the castOp
      if (!isa<top::NoneOp>(module::getOriValue(
               op.getOperand(i+1)).getDefiningOp())) {
        auto yield_type = yieldOp->getOperand(i).getType();
        auto opt = op.getOperand(i+1).getType();

        if ((module::isCalibratedType(yield_type) != module::isCalibratedType(opt))
           || (module::isUniformQuantized(yield_type) != module::isUniformQuantized(opt))
           || (cast<RankedTensorType>(yield_type).getElementTypeBitWidth()
            != cast<RankedTensorType>(opt).getElementTypeBitWidth())) {
          auto loc = module::getLocLike(module::getOriValue
                               (yieldOp->getOperand(i)), std::to_string(i));
          auto castOp = builder.create<tpu::CastOp>(loc, opt,
                                ValueRange{yieldOp->getOperand(i)});

          yieldOp->getOperand(i).replaceUsesWithIf(castOp.getOutput(),
                                             [&](OpOperand &use) {
                                               return isa<tpu::YieldOp>(use.getOwner());});
        }
      }
    }

    //update the loopop's output
    for (int i = 0; i < op.v_final().size(); i++) {
      auto type = yieldOp->getOperand(i+1).getType();
      op.getResult(i).setType(type);
    }
    return success();
  }
};

void tpu::LoopOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<UpdateArgument>(context);
}

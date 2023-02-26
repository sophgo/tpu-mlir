//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTosa/OpLowering.h"
#include "tpu_mlir/Conversion/TopToTosa/TopToTosa.h"
#include "tpu_mlir/Support/Module.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOPTOTOSA
#include "tpu_mlir/Conversion/Passes.h.inc"
} // namespace mlir

namespace tpu_mlir {

struct EraseTopInputOp : public OpRewritePattern<top::InputOp> {
  using OpRewritePattern<top::InputOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(top::InputOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertTopToTosa
    : public ::impl::ConvertTopToTosaBase<ConvertTopToTosa> {
public:
  void runOnOperation() override {
    module_ = getOperation();
    ctx_ = &getContext();
    mainFunc_ = module::getMainFuncOp();

    RewritePatternSet patterns(ctx_);
    ConversionTarget target(*ctx_);
    target.addLegalDialect<mlir::tosa::TosaDialect>();
    populateTopToTosaConversionPatterns(&patterns);

    auto config = GreedyRewriteConfig();
    config.maxIterations = 0; // apply each pattern only once.
    applyPatternsAndFoldGreedily(module_, std::move(patterns), config);

    patterns.clear();
    patterns.add<EraseTopInputOp>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    module::updateModuleTypes();
    module::setState(module::State::TOSA_F32);
  }

protected:
  ModuleOp module_;
  FuncOp mainFunc_;
  MLIRContext *ctx_;
};

std::unique_ptr<Pass> createConvertTopToTosa() {
  return std::make_unique<ConvertTopToTosa>();
}

} // namespace tpu_mlir

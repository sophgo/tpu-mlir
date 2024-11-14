#include "Passes.h"
namespace mlir {

/// Pattern to convert a tensor.tensor operation into a fill +
/// tensor.insert_slice. This is needed till tensor.pad op can be fused with its
/// consumers.
struct TensorPadOpConversion : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;
  TensorPadOpConversion(MLIRContext *context, bool skipSingleLinalgOpUses)
      : OpRewritePattern<tensor::PadOp>(context, skipSingleLinalgOpUses),
        skipSingleLinalgOpUses(skipSingleLinalgOpUses) {}

  LogicalResult matchAndRewrite(tensor::PadOp padTensorOp,
                                PatternRewriter &rewriter) const override {
    // Check that the region is just a yield operation which is returning a
    // scalar that is not one of the arguments of the linalg operation.
    Region &region = padTensorOp.getRegion();
    Block &block = region.front();
    if (!llvm::hasSingleElement(block))
      return failure();
    auto yieldOp = cast<tensor::YieldOp>(block.getTerminator());
    Value yieldVal = yieldOp.getValue();
    if (llvm::any_of(block.getArguments(),
                     [&](Value v) { return v == yieldVal; })) {
      return failure();
    }

    if (padTensorOp->hasOneUse()) {
      Operation *use = padTensorOp->use_begin()->getOwner();
      if (skipSingleLinalgOpUses) {
        if (isa<linalg::LinalgOp>(use) &&
            !isa<linalg::Conv2DNhwcHwcfQOp, linalg::DepthwiseConv2DNhwcHwcQOp,
                 linalg::DepthwiseConv2DNhwcHwcmQOp>(use)) {
          return failure();
        }
      }
    }

    // Rewrite tensor.pad to tensor.empty + linalg.fill + tensor.insert_slice.
    return static_cast<LogicalResult>(
        linalg::rewriteInDestinationPassingStyle(rewriter, padTensorOp));
  }

private:
  // Option to skip the pattern when tensor.pad op has one use and is used by
  // a Linalg op.
  bool skipSingleLinalgOpUses = false;
};

struct TensorPadToTensorInsertSlicePass
    : public PassWrapper<TensorPadToTensorInsertSlicePass,
                         OperationPass<ModuleOp>> {
  TensorPadToTensorInsertSlicePass(bool skipSingleLinalgOpUses)
      : skipSingleLinalgOpUses(skipSingleLinalgOpUses) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, memref::MemRefDialect, func::FuncDialect,
                mlir::math::MathDialect, mlir::arith::ArithDialect>();
  }

  llvm::StringRef getArgument() const override {
    return "tensor-pad-to-tensor-insert-slice";
  }

  llvm::StringRef getDescription() const override {
    return "Convert tensor.pad into linalg.fill + tensor.insert_slice";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<TensorPadOpConversion>(context, skipSingleLinalgOpUses);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

private:
  bool skipSingleLinalgOpUses;
};

std::unique_ptr<OperationPass<ModuleOp>>
createTensorPadToTensorInsertSlicePass(bool skipSingleLinalgOpUses) {
  return std::make_unique<TensorPadToTensorInsertSlicePass>(
      skipSingleLinalgOpUses);
}
} // namespace mlir

#include "Passes.h"

namespace mlir {
struct EraseUnusedLinalgOperandsPass
    : public EraseUnusedLinalgOperandsBase<EraseUnusedLinalgOperandsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    linalg::populateEraseUnnecessaryInputsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createEraseUnusedLinalgOperands() {
  return std::make_unique<EraseUnusedLinalgOperandsPass>();
}
} // namespace mlir

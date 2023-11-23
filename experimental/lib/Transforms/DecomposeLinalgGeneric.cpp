//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu-mlir/Transforms/Passes.h"

namespace tpu_mlir {
using namespace mlir;

class DecomposeLinalgGenericPass
    : public DecomposeLinalgGenericBase<DecomposeLinalgGenericPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    linalg::populateDecomposeLinalgOpsPattern(patterns);
    linalg::GenericOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createDecomposeLinalgGenericPass() {
  return std::make_unique<DecomposeLinalgGenericPass>();
}

std::unique_ptr<Pass> createBufferizePass() {
  return std::make_unique<DecomposeLinalgGenericPass>();
}

} // namespace tpu_mlir

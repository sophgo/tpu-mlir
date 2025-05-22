//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

extern void
populateOptPostProcessorBM1684XPatterns(RewritePatternSet *patterns);

class OptPostProcessorPass : public OptPostProcessorBase<OptPostProcessorPass> {
public:
  OptPostProcessorPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    RewritePatternSet patterns(mOp.getContext());
    if (module::isMARS3()) {
      populateOptPostProcessorBM1684XPatterns(&patterns);
    }
    applyPatternsAndFoldGreedily(mOp, std::move(patterns));
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createOptPostProcessorPass() {
  return std::make_unique<OptPostProcessorPass>();
}
} // namespace tpu
} // namespace tpu_mlir

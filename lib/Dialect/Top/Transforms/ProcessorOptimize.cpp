//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"

using namespace llvm;

namespace tpu_mlir {
namespace top {
extern void populateOptimizeBM1684XPatterns(RewritePatternSet *patterns);
extern void populateOptimizeBM1684Patterns(RewritePatternSet *patterns);
extern void populateOptimizeCV18XXPatterns(RewritePatternSet *patterns);

class ProcessorOptimizePass
    : public ProcessorOptimizeBase<ProcessorOptimizePass> {
public:
  ProcessorOptimizePass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    RewritePatternSet patterns(mOp.getContext());
    if (module::isBM1684Family()) {
      populateOptimizeBM1684Patterns(&patterns);
    } else if (module::isCV18xx()) {
      populateOptimizeCV18XXPatterns(&patterns);
    } else {
      // 1684x as default
      populateOptimizeBM1684XPatterns(&patterns);
    }
    applyPatternsAndFoldGreedily(mOp, std::move(patterns));
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createProcessorOptimizePass() {
  return std::make_unique<ProcessorOptimizePass>();
}
} // namespace top
} // namespace tpu_mlir

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

extern void populateOptimizeBM1684Patterns(RewritePatternSet *patterns);
extern void populateOptimizeBM1684XPatterns(RewritePatternSet *patterns);
extern void populateOptimizeCV18XXPatterns(RewritePatternSet *patterns);

class ProcessorOptimizePass
    : public ProcessorOptimizeBase<ProcessorOptimizePass> {
public:
  ProcessorOptimizePass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    RewritePatternSet patterns(mOp.getContext());
    if (module::isBM1684XFamily() || module::isBM1690Family()) {
      populateOptimizeBM1684XPatterns(&patterns);
    } else if (module::isCV18xx()) {
      populateOptimizeCV18XXPatterns(&patterns);
    } else if (module::isBM1684Family()) {
      populateOptimizeBM1684Patterns(&patterns);
    }
    applyPatternsAndFoldGreedily(mOp, std::move(patterns));
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createProcessorOptimizePass() {
  return std::make_unique<ProcessorOptimizePass>();
}
} // namespace tpu
} // namespace tpu_mlir

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;


namespace tpu_mlir {
namespace tpu {

extern void populateOptimizeBM1684Patterns(RewritePatternSet *patterns);
extern void populateOptimizeBM1684XPatterns(RewritePatternSet *patterns);
extern void populateOptimizeCV18XXPatterns(RewritePatternSet *patterns);
extern void populateOptimizeBM168xPatterns(RewritePatternSet *patterns);

class ChipOptimizePass : public ChipOptimizeBase<ChipOptimizePass> {
public:
  ChipOptimizePass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    RewritePatternSet patterns(mOp.getContext());
    if (module::isBM1684XFamily()) {
      populateOptimizeBM168xPatterns(&patterns);
      populateOptimizeBM1684XPatterns(&patterns);
    } else if (module::isCV18xx()) {
      populateOptimizeCV18XXPatterns(&patterns);
    } else if (module::isBM1684Family()) {
      populateOptimizeBM168xPatterns(&patterns);
      populateOptimizeBM1684Patterns(&patterns);
    }
    applyPatternsAndFoldGreedily(mOp, std::move(patterns));
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createChipOptimizePass() {
  return std::make_unique<ChipOptimizePass>();
}
} // namespace tpu
} // namespace tpu_mlir

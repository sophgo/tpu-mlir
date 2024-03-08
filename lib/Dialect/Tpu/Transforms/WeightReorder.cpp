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
#include "tpu_mlir/Backend/Arch.h"

using namespace llvm;

using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {
extern void populateWeightReorderCV18xxPatterns(RewritePatternSet *patterns);
extern void populateWeightReorderBM1684Patterns(RewritePatternSet *patterns);
extern void populateWeightReorderBM1684XPatterns(RewritePatternSet *patterns);

class WeightReorderPass : public WeightReorderBase<WeightReorderPass> {
public:
  WeightReorderPass() {}
  void runOnOperation() override {
    if (!module::isState(module::State::TPU_LOWERED)) {
      llvm_unreachable("module should be tpu quantized");
    }
    auto modules = module::getAllModules();
    for (auto sub : *modules) {
      RewritePatternSet patterns(&getContext());
      if (module::isBM1684Family()) {
        populateWeightReorderBM1684Patterns(&patterns);
      } else if (module::isBM1684XFamily() || module::isBM1690Family()) {
        populateWeightReorderBM1684XPatterns(&patterns);
      } else if (module::isCV18xx()) {
        populateWeightReorderCV18xxPatterns(&patterns);
      }
      auto config = GreedyRewriteConfig();
      config.maxIterations = 1; // apply each pattern only once.
      applyPatternsAndFoldGreedily(sub, std::move(patterns), config);
    }
    module::updateModuleTypes();
    module::setState(module::State::TPU_REORDERED);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createWeightReorderPass() {
  return std::make_unique<WeightReorderPass>();
}
} // namespace tpu
} // namespace tpu_mlir

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
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/RewriterConfigUtils.h"

#define CONFIG_FILE_NAME                                                       \
  module::getName(module::getModuleOp()).str() + "_" +                         \
      module::getChipStr().str() + "_" + module::getModeStr() +                \
      ".tpu_processor_optimize.json"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

extern void populateOptimizeBM1684Patterns(RewritePatternSet *patterns);
extern void
populateOptimizeBM1684XPatterns(RewritePatternSet *patterns,
                                const std::vector<RewriterRule> &rules);
extern void populateOptimizeCV18XXPatterns(RewritePatternSet *patterns);

class ProcessorOptimizePass
    : public ProcessorOptimizeBase<ProcessorOptimizePass> {
public:
  ProcessorOptimizePass() {}
  void runOnOperation() override {
    // load rewrite rules from config file
    std::string configPath = CONFIG_FILE_NAME;
    if (getenv("TPU_DIALECT_REWRITER_CONFIG")) {
      configPath = std::string(getenv("TPU_DIALECT_REWRITER_CONFIG"));
    }
    auto rules = loadRewriteConfig(configPath);
    PASS_LOG_DEBUG_BLOCK({ dumpRewriterRules(rules, llvm::outs(), true); });

    auto mOp = getOperation();
    RewritePatternSet patterns(mOp.getContext());
    if (module::isBM1684XFamily() || module::isBM1690Family()) {
      populateOptimizeBM1684XPatterns(&patterns, rules);
    } else if (module::isCV18xx()) {
      populateOptimizeCV18XXPatterns(&patterns);
    } else if (module::isBM1684Family()) {
      populateOptimizeBM1684Patterns(&patterns);
    }
    applyPatternsAndFoldGreedily(mOp, std::move(patterns));

    // set multicore flag if support
    if (module::getCoreNum() > 1) {
      mOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (false == module::isOpInBlock(op)) {
          auto gl = dyn_cast<GlobalGenInterface>(op);
          if (gl && gl.support_multi_core()) {
            mlir::Attribute isTrue =
                mlir::BoolAttr::get(op->getContext(), true);
            op->setAttr("multicore", isTrue);
          }
        }
      });
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createProcessorOptimizePass() {
  return std::make_unique<ProcessorOptimizePass>();
}
} // namespace tpu
} // namespace tpu_mlir

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
// #include "tpu_mlir/Dialect/Tpu/Transforms/BM1684/ChipOptimize.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/ChipOptimize.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/ChipOptimize.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/Module.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "tpu_mlir/Backend/Arch.h"

#include <cstdint>
#include <fstream>
#include <set>
#include <sstream>

using namespace llvm;
using namespace mlir;

using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {

class ChipOptimizePass : public ChipOptimizeBase<ChipOptimizePass> {
public:
  ChipOptimizePass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    RewritePatternSet patterns(mOp.getContext());
    if (module::isBM1684XFamily()) {
      bm1684x::populateChipOptimizePatterns(&patterns);
    } else if (module::isCV18xx()) {
      cv18xx::populateChipOptimizePatterns(&patterns);
    } else if (module::isBM1684Family()) {
      bm1684::populateChipOptimizePatterns(&patterns);
    }
    auto config = GreedyRewriteConfig();
    config.maxIterations = 5;
    applyPatternsAndFoldGreedily(mOp, std::move(patterns), config);
    module::updateModuleTypes();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createChipOptimizePass() {
  return std::make_unique<ChipOptimizePass>();
}
} // namespace tpu
} // namespace tpu_mlir

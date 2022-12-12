//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM1684/WeightReorder.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/WeightReorder.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/WeightReorder.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/Module.h"
#include "mlir/IR/BlockAndValueMapping.h"
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

class WeightReorderPass : public WeightReorderBase<WeightReorderPass> {
public:
  WeightReorderPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    if (!module::isState(module::State::TPU_LOWERED)) {
      llvm_unreachable("module should be tpu quantized");
    }
    Arch::init();
    RewritePatternSet patterns(mOp.getContext());
    if (module::isBM1684Family()) {
      bm1684::populateWeightReorderPatterns(&patterns);
    } else if (module::isBM1684XFamily()) {
      bm1684x::populateWeightReorderPatterns(&patterns);
    } else if (module::isCV18xx()) {
      cv18xx::populateWeightReorderPatterns(&patterns);
    }
    auto config = GreedyRewriteConfig();
    config.maxIterations = 0; // apply each pattern only once.
    applyPatternsAndFoldGreedily(mOp, std::move(patterns), config);
    module::updateModuleTypes();
    module::setState(module::State::TPU_REORDERED);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createWeightReorderPass() {
  return std::make_unique<WeightReorderPass>();
}
} // namespace tpu
} // namespace tpu_mlir

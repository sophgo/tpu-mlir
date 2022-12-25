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
#include "tpu_mlir/Support/Helper/Module.h"

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <fstream>
#include <set>
#include <sstream>

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {

class WeightReorderPass : public WeightReorderBase<WeightReorderPass> {
public:
  WeightReorderPass() {}
  void runOnOperation() override {
    auto module = getOperation();
    if (!Module::isState(Module::State::TPU_LOWERED)) {
      llvm_unreachable("module should be tpu quantized");
    }
    Arch::init();
    RewritePatternSet patterns(module.getContext());
    if (Module::isBM1684Family()) {
      bm1684::populateWeightReorderPatterns(&patterns);
    } else if (Module::isBM1684XFamily()) {
      bm1684x::populateWeightReorderPatterns(&patterns);
    } else if (Module::isCV18xx()) {
      cv18xx::populateWeightReorderPatterns(&patterns);
    }
    auto config = GreedyRewriteConfig();
    config.maxIterations = 0; // apply each pattern only once.
    applyPatternsAndFoldGreedily(module, std::move(patterns), config);
    Module::updateModuleTypes();
    Module::setState(Module::State::TPU_REORDERED);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createWeightReorderPass() {
  return std::make_unique<WeightReorderPass>();
}
} // namespace tpu
} // namespace tpu_mlir

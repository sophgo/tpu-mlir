//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
// #include "tpu_mlir/Dialect/Tpu/Transforms/BM1684/DoExtraOpt.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DoExtraOpt.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/DoExtraOpt.h"
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

class DoExtraOptPass : public DoExtraOptBase<DoExtraOptPass> {
public:
  DoExtraOptPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    RewritePatternSet patterns(mOp.getContext());
    if (module::isBM1684XFamily()) {
      bm1684x::populateDoExtraOptPatterns(&patterns);
    } else if (module::isCV18xx()) {
      cv18xx::populateDoExtraOptPatterns(&patterns);
    }
    auto config = GreedyRewriteConfig();
    config.maxIterations = 5; // apply each pattern only once.
    applyPatternsAndFoldGreedily(mOp, std::move(patterns), config);
    module::updateModuleTypes();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createDoExtraOptPass() {
  return std::make_unique<DoExtraOptPass>();
}
} // namespace tpu
} // namespace tpu_mlir

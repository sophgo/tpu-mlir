//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Support/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;

namespace tpu_mlir {
namespace top {

class DeinitPass : public DeinitBase<DeinitPass> {
public:
  DeinitPass() {}
  void runOnOperation() override {
    auto state = module::getState();
    if (state >= module::State::TOSA_F32) {
      return;
    }
    module::removeUnusedOp();
    module::saveWeight();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createDeinitPass() {
  return std::make_unique<DeinitPass>();
}
} // namespace top
} // namespace tpu_mlir

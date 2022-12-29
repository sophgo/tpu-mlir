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

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <set>

using namespace llvm;
using namespace mlir;

namespace tpu_mlir {
namespace top {

class InitPass : public InitBase<InitPass> {
public:
  InitPass() {}
  void runOnOperation() override {
    auto module = getOperation();
    module::init(module);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createInitPass() {
  return std::make_unique<InitPass>();
}
} // namespace top
} // namespace tpu_mlir

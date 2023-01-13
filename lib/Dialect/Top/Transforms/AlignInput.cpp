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
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;

namespace tpu_mlir {
namespace top {

class AlignInputPassPass
    : public AlignInputBase<AlignInputPassPass> {
public:
  AlignInputPassPass() {}
  void runOnOperation() override {
    llvm::errs()<<"Entering AlignInputPass,todo.\n";
  }
};
std::unique_ptr<OperationPass<ModuleOp>> createAlignInputPass() {
  return std::make_unique<AlignInputPassPass>();
}
}
}

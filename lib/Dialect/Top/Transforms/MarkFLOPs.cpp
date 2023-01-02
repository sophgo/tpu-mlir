//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"

using namespace llvm;
using namespace mlir;

namespace tpu_mlir {
namespace top {

class MarkFLOPsPass : public MarkFLOPsBase<MarkFLOPsPass> {
public:
  MarkFLOPsPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    int64_t flops = 0;
    for (auto func : mOp.getOps<FuncOp>()) {
      func.walk([&](FlopsInterface op) { flops += op.getFLOPs(); });
    }
    module::setFLOPs(flops);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createMarkFLOPsPass() {
  return std::make_unique<MarkFLOPsPass>();
}
} // namespace top
} // namespace tpu_mlir

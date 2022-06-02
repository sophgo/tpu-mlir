//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir::helper;
namespace tpu_mlir {
namespace top {

class MarkFLOPsPass : public MarkFLOPsBase<MarkFLOPsPass> {
public:
  MarkFLOPsPass() {}
  void runOnOperation() override {
    auto module = getOperation();
    int64_t flops = 0;
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](FlopsInterface op) { flops += op.getFLOPs(); });
    }
    Module::setFLOPs(module, flops);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createMarkFLOPsPass() {
  return std::make_unique<MarkFLOPsPass>();
}
} // namespace top
} // namespace tpu_mlir

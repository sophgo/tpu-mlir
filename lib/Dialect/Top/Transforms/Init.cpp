//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/Arch.h"
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"

using namespace llvm;

namespace tpu_mlir {
int32_t cur_log_level = 0;
namespace top {

class InitPass : public InitBase<InitPass> {
public:
  InitPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    module::init(mOp);
    backend::Arch::init(freq);
    module::init_loglevel(this->level);
    module::setWeightInMemFlag(weight_in_mem);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createInitPass() {
  return std::make_unique<InitPass>();
}
} // namespace top
} // namespace tpu_mlir

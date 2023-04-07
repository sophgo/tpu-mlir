//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/Module.h"
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

namespace tpu_mlir {
namespace tpu {

class ChipAssignPass : public ChipAssignBase<ChipAssignPass> {
public:
  ChipAssignPass() {}
  void runOnOperation() override {
    auto chip_ = StringRef(type).lower();
    auto chip = module::symbolizeChip(chip_);
    assert(chip.has_value());
    module::setChip(chip.value());
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createChipAssignPass() {
  return std::make_unique<ChipAssignPass>();
}
} // namespace tpu
} // namespace tpu_mlir

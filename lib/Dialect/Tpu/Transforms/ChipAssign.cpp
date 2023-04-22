//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//


#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/Module.h"
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

    // for cv18xx , input only support fp32
    if (module::isCV18xx()) {
      input_type_process();
    }
    module::updateModuleTypes();
  }

  void input_type_process() {
    auto mainFunc = module::getMainFuncOp();
    mainFunc.walk([&](Operation *op) {
      if (isa<top::InputOp>(op)) {
        auto output_value = op->getResult(0);
        auto storage_type = module::getStorageType(output_value);
        if (storage_type.isIntOrIndex()) {
          auto new_type = RankedTensorType::get(module::getShape(output_value),
                                                Builder(op).getF32Type());
          output_value.setType(new_type);
        }
      }
    });
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createChipAssignPass() {
  return std::make_unique<ChipAssignPass>();
}
} // namespace tpu
} // namespace tpu_mlir

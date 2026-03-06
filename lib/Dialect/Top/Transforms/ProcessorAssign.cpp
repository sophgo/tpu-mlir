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

using namespace tpu_mlir::backend;
using namespace llvm;

namespace tpu_mlir {
namespace top {

class ProcessorAssignPass : public ProcessorAssignBase<ProcessorAssignPass> {
public:
  ProcessorAssignPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    auto chip_ = StringRef(chip).lower();
    auto chip = module::symbolizeChip(chip_);
    assert(chip.has_value());
    module::setChip(chip.value());
    if (!(module::isBM1684XFamily() || module::isBM1690Family())) {
      // only one device
      num_device = 1;
    }
    if (!(module::isBM1688() || module::isBM1690Family() ||
          module::isSG2380()) ||
        chip_ == "cv186x") {
      // only one core
      num_core = 1;
    }
    auto mode_ = StringRef(mode).upper();
    auto quant_mode = module::symbolizeMode(mode_);
    assert(quant_mode.has_value());
    module::setMode(quant_mode.value());
    assert(num_device > 0);
    module::setDeviceNum(num_device);
    assert(num_core > 0);
    module::setCoreNum(num_core);
    auto mode = module::AddrMode::BASIC;
    if (addr_mode != "auto") {
      mode = module::symbolizeAddrMode(addr_mode).value_or(
          module::AddrMode::BASIC);
    }
    module::setAddrMode(mode);
    module::setHighPrecision(high_precision);
    backend::Arch::init(0);
    // for cv18xx , input only support fp32
    if (module::isCV18xx()) {
      input_type_process(mOp);
    }
    module::updateModuleTypes();
  }

private:
  void input_type_process(ModuleOp mOp) {
    auto mainFunc = module::getMainFuncOp(mOp);
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

std::unique_ptr<OperationPass<ModuleOp>> createProcessorAssignPass() {
  return std::make_unique<ProcessorAssignPass>();
}
} // namespace top
} // namespace tpu_mlir

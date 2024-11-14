//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

class ShowAddressPass : public ShowAddressBase<ShowAddressPass> {
public:
  ShowAddressPass() {}
  void runOnOperation() override {
    if (!module::isState(module::State::TPU_ADDRESSED)) {
      llvm_unreachable("module should be addressed");
    }
    auto modules = module::getAllModules();
    for (auto s : *modules) {
      auto name = module::getName(s);
      int64_t coeff_addr = 0, neuron_addr = 0;
      if (!module::isCV18xx()) {
        coeff_addr = module::getCoeffAddr(s);
        neuron_addr = module::getNeuronAddr(s);
      }
      auto coeff_size = module::getCoeffSize(s);
      auto neuron_size = module::getNeuronSize(s);
      printf("\n=================================================\n");
      printf("Net name: %s\n", name.str().c_str());
      printf("Coeff size: 0x%lX\n", coeff_size);
      printf("Neuron size: 0x%lX\n", neuron_size);
      printf("-------------- coeff addr ----------------------\n");
      for (auto func : s.getOps<FuncOp>()) {
        func.walk([&](top::WeightOp op) {
          auto addr = module::getAddress(op.getOutput());
          auto size = module::getBytes(op.getOutput());
          auto name = module::getName(op.getOutput());
          printf("[%s] : addr[0x%lX, 0x%lX), size[0x%lX]\n", name.str().c_str(),
                 addr - coeff_addr, addr + size - coeff_addr, size);
        });
      }
      printf("-------------- neuron addr ----------------------\n");
      for (auto func : s.getOps<FuncOp>()) {
        func.walk([&](Operation *op) {
          if (isa<func::FuncOp, func::ReturnOp, func::CallOp, top::WeightOp,
                  top::NoneOp>(op)) {
          } else if (module::isOpInBlock(op)) {
          } else {
            for (auto out : op->getResults()) {
              if (module::isNone(out)) {
                continue;
              }
              auto addr = module::getAddress(out);
              auto size = module::getBytes(out);
              auto name = module::getName(out);
              printf("[%s] : addr[0x%lX, 0x%lX), size[0x%lX]\n",
                     name.str().c_str(), addr - neuron_addr,
                     addr + size - neuron_addr, size);
            }
          }
        });
      }
      printf("=================================================\n");
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createShowAddressPass() {
  return std::make_unique<ShowAddressPass>();
}
} // namespace tpu
} // namespace tpu_mlir

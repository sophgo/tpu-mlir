#include "sophgo/Dialect/Tpu/Transforms/Passes.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Helper/Module.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <sstream>
#include <fstream>
#include <set>

using namespace llvm;
using namespace mlir;
using namespace sophgo::helper;
namespace sophgo {
namespace tpu {

class WeightReorderPass : public WeightReorderBase<WeightReorderPass> {
public:
  WeightReorderPass() {}
  void runOnOperation() override {
    auto module = getOperation();
    auto state = Module::getState(module);
    if (state != Module::State::TPU_LOWERED) {
      llvm_unreachable("module should be tpu quantized");
    }
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](WeightReorderInterface op) {
        op.weight_reorder();
      });
    }
    Module::setState(module, Module::State::TPU_REORDERED);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createWeightReorderPass() {
  return std::make_unique<WeightReorderPass>();
}
} // namespace top
} // namespace sophgo

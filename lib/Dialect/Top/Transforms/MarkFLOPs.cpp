#include "sophgo/Dialect/Top/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"

using namespace llvm;
using namespace mlir;
using namespace sophgo::helper;
namespace sophgo {
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
} // namespace sophgo

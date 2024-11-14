#include "Passes.h"

namespace mlir {
class StripDebugOpPass
    : public PassWrapper<StripDebugOpPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override { return "strip-debugop"; }

  StringRef getDescription() const override {
    return "erase the debug op(such as cf.assert)";
  }

  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      if (isa<mlir::cf::AssertOp>(op))
        op->erase();
    });
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createStripDebugOpPass() {
  return std::make_unique<StripDebugOpPass>();
}

static PassRegistration<StripDebugOpPass> pass([] {
  return std::make_unique<StripDebugOpPass>();
});
} // namespace mlir

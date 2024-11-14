#include "include/Utils.h"

namespace mlir {
class SetEntryPointPass
    : public PassWrapper<SetEntryPointPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override { return "set-entry-point"; }

  StringRef getDescription() const override { return "set the entry point"; }

  void runOnOperation() override {
    for (auto it :
         llvm::enumerate(getOperation().getOps<FunctionOpInterface>())) {
      FunctionOpInterface op = it.value();
      Operation *operation = op;
      if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(operation)) {
        std::string name = funcOp.getName().str();
        if (name.compare("main") && funcOp.isPublic() &&
            !funcOp.isDeclaration()) {
          funcOp.setName("main");
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createSetEntryPointPass() {
  return std::make_unique<SetEntryPointPass>();
}

static PassRegistration<SetEntryPointPass> pass;
} // namespace mlir

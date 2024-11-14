#include "include/Utils.h"

namespace mlir {
class AssignTargetDevicePass
    : public PassWrapper<AssignTargetDevicePass, OperationPass<ModuleOp>> {
public:
  AssignTargetDevicePass() : target("BM1684X") {}
  AssignTargetDevicePass(std::string target_) : target(std::move(target_)) {}

  StringRef getArgument() const override { return "assign-target-device"; }

  StringRef getDescription() const override {
    return "Assigns the devices the module will target to the given list "
           "of targets.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Check to see if targets are already specified.
    auto existingTargetsAttr =
        moduleOp->getAttrOfType<StringAttr>("device.target");
    if (existingTargetsAttr) {
      // Targets already exist on the module; no-op the pass so that we don't
      // mess with whatever the user intended.
      return;
    }

    // If no targets are specified we can't do anything - another pass earlier
    // in the pipeline will have had to add the targets.
    if (target.empty()) {
      emitRemark(moduleOp.getLoc())
          << "no target HAL devices specified during assignment";
      return;
    }
    moduleOp->setAttr("device.target",
                      StringAttr::get(moduleOp.getContext(), target));
  }

private:
  std::string target;
};

std::unique_ptr<OperationPass<ModuleOp>>
createAssignTargetDevicePass(std::string targets) {
  return std::make_unique<AssignTargetDevicePass>(targets);
}

static PassRegistration<AssignTargetDevicePass> pass([] {
  return std::make_unique<AssignTargetDevicePass>();
});
} // namespace mlir

#include "Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
namespace mlir {
// make sure the tosa/stablehlo dialect have lowered to linalg-on-tensor before
// subgraph split pipeline
class VerifyInputLegalityPass
    : public PassWrapper<VerifyInputLegalityPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override { return "verify-input-legality"; }

  StringRef getDescription() const override {
    return "Checks the legality of the IR at the start of subgraph split "
           "transformation pipeline";
  }

  static void emitLegalizationErrors(Location loc,
                                     const DenseSet<Operation *> &illegalOps) {
    // Print op errors for each of the illegal ops that still remain.
    llvm::MapVector<StringRef, int> opNameCounts;
    for (Operation *illegalOp : illegalOps) {
      StringRef opName = illegalOp->getName().getStringRef();
      opNameCounts[opName]++;
      illegalOp->emitOpError() << ": illegal op still exists";
    }

    std::vector<std::string> errorMessages;
    errorMessages.reserve(opNameCounts.size());
    for (const auto &opInfo : opNameCounts) {
      errorMessages.push_back(
          llvm::formatv("\t{0} (count: {1})", opInfo.first, opInfo.second));
    }
    emitError(loc) << "The following illegal operations still remain: \n"
                   << llvm::join(errorMessages, "\n") << "\n";
  }

  LogicalResult verifyAllOperationsAreLegal(Operation *op,
                                            const ConversionTarget &target) {
    // We don't just use applyPartialConversion with no patterns because this
    // pass shouldn't alter the IR at all (including via folding or
    // canonicalizations that dialect conversion does automatically).
    DenseSet<Operation *> illegalOps;
    op->walk([&](Operation *op) {
      if (!target.isLegal(op)) {
        illegalOps.insert(op);
      }
    });
    if (illegalOps.empty())
      return success();
    emitLegalizationErrors(op->getLoc(), illegalOps);
    return failure();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addLegalOp<tosa::ApplyScaleOp>();
    // We're already depending on the Tosa Dialect
    target.addIllegalDialect<tosa::TosaDialect>();
    // Avoid StableHLO dependency
    target.addIllegalDialect("chlo");
    target.addIllegalDialect("stablehlo");
    target.addIllegalOp<UnrealizedConversionCastOp>();

    if (failed(verifyAllOperationsAreLegal(getOperation(), target))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createVerifyInputLegalityPass() {
  return std::make_unique<VerifyInputLegalityPass>();
}

static PassRegistration<VerifyInputLegalityPass> pass([] {
  return std::make_unique<VerifyInputLegalityPass>();
});
} // namespace  mlir

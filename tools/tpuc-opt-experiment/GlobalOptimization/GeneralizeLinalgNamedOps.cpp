#include "Passes.h"

namespace mlir {
// convert some linalg named op to linalg genericOp for later tile + fuse
struct GeneralizeLinalgNamedOpsPass
    : public GeneralizeLinalgNamedOpsBase<GeneralizeLinalgNamedOpsPass> {

  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<linalg::LinalgOp> namedOpCandidates;
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      if (isa_and_nonnull<linalg::AbsOp, linalg::AddOp, linalg::BroadcastOp,
                          linalg::CeilOp, linalg::CopyOp, linalg::DivOp,
                          linalg::ElemwiseBinaryOp, linalg::ElemwiseUnaryOp,
                          linalg::ExpOp, linalg::FloorOp, linalg::LogOp,
                          linalg::MapOp, linalg::MaxOp, linalg::MulOp,
                          linalg::ReduceOp, linalg::SubOp, linalg::TransposeOp>(
              linalgOp.getOperation())) {
        namedOpCandidates.push_back(linalgOp);
      }
    });

    IRRewriter rewriter(&getContext());
    for (auto linalgOp : namedOpCandidates) {
      rewriter.setInsertionPoint(linalgOp);
      FailureOr<linalg::GenericOp> generalizedOp =
          linalg::generalizeNamedOp(rewriter, linalgOp);
      if (failed(generalizedOp)) {
        linalgOp->emitOpError("failed to generalize operation");
        return signalPassFailure();
      }
    }
  }
};

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGeneralizeLinalgNamedOpsPass() {
  return std::make_unique<GeneralizeLinalgNamedOpsPass>();
}
} // namespace mlir

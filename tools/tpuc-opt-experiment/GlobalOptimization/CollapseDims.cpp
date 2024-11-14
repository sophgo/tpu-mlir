#include "Passes.h"
namespace mlir {

static bool hasContiguousDims(AffineMap map, ArrayRef<unsigned> dims) {
  if (!map.isProjectedPermutation())
    return false;
  llvm::SmallDenseSet<unsigned> existingDims(dims.begin(), dims.end());
  for (unsigned i = 0, e = map.getNumResults(); i < e; i++) {
    if (map.getDimPosition(i) != dims[0]) {
      if (existingDims.count(map.getDimPosition(i))) {
        return false;
      }
      continue;
    }
    // Check that the following dimensions are match the order of `dims`
    for (unsigned j = 1, numDims = dims.size(); j < numDims; j++) {
      unsigned pos = i + j;
      if (pos >= map.getNumResults() || map.getDimPosition(pos) != dims[j]) {
        return false;
      }
    }
    break;
  }
  return true;
}

struct CollapseDimsPass
    : public PassWrapper<CollapseDimsPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  StringRef getArgument() const override { return "collapse-dims"; }

  StringRef getDescription() const override {
    return "Collapse reduction dimensions when possible";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    linalg::GetCollapsableDimensionsFn collapseFn =
        [&](linalg::GenericOp op) -> SmallVector<ReassociationIndices> {
      SmallVector<ReassociationIndices> collapseIndices;
      SmallVector<unsigned> reductionDims;
      op.getReductionDims(reductionDims);
      if (reductionDims.size() < 2)
        return collapseIndices;

      for (AffineMap map : op.getIndexingMapsArray()) {
        if (!hasContiguousDims(map, reductionDims))
          return collapseIndices;
      }
      ReassociationIndices indices;
      for (unsigned dim : reductionDims) {
        indices.push_back(int64_t(dim));
      }
      collapseIndices.push_back(indices);
      return collapseIndices;
    };

    linalg::populateCollapseDimensions(patterns, collapseFn);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createCollapseDimsPass() {
  return std::make_unique<CollapseDimsPass>();
}
} // namespace mlir

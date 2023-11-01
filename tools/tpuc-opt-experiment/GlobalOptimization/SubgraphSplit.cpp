#include "Passes.h"

namespace mlir
{

class SubgraphSplitPass
  : public PassWrapper<SubgraphSplitPass, mlir::InterfacePass<mlir::FunctionOpInterface>>
{
private:
  /* data */
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    mlir::MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    {
      //legalize the control flow op
      scf::IfOp::getCanonicalizationPatterns(patterns, context);
      scf::ExecuteRegionOp::getCanonicalizationPatterns(patterns, context);
      scf::ForOp::getCanonicalizationPatterns(patterns, context);
      scf::ForallOp::getCanonicalizationPatterns(patterns, context);
      scf::ParallelOp::getCanonicalizationPatterns(patterns, context);
      scf::WhileOp::getCanonicalizationPatterns(patterns, context);
      affine::AffineIfOp::getCanonicalizationPatterns(patterns, context);
      if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    //Todo: subgraph split
  }
};

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createSubgraphSplitPass() {
  return std::make_unique<SubgraphSplitPass>();
}
}


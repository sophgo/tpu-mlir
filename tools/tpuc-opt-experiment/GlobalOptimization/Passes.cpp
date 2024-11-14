#include "Passes.h"
namespace mlir {

using FunctionLikeNest = MultiOpNest<mlir::func::FuncOp>;
void buildGlobalOptimizationPassPipeline(OpPassManager &pm, bool dynamic_mode) {
  // Preprocessing passes to get the program into a canonical state.
  FunctionLikeNest(pm)
      .addPass(mlir::createRemoveZeroExtentTensorsPass)
      .addPass(mlir::createDetachElementwiseFromNamedOpsPass)
      .addPass(mlir::createLinalgNamedOpConversionPass)
      .addPass(mlir::createConvert1X1FilterConv2DToMatmulPass);
  pm.addPass(mlir::createEraseUnusedLinalgOperands());

  FunctionLikeNest(pm)
      .addPass(mlir::createConvertElementwiseToLinalgPass)
      .addPass(mlir::createGeneralizeLinalgNamedOpsPass)
      .addPass(mlir::createLinalgFoldUnitExtentDimsPass)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)
      .addPass(mlir::createStripDebugOpPass);
  pm.addPass(mlir::createVerifyInputLegalityPass());
  pm.addPass(mlir::createSymbolDCEPass());

  // Transform pad operations into linalg.fill + tensor.insert_slice
  pm.addPass(mlir::createTensorPadToTensorInsertSlicePass());

  FunctionLikeNest(pm)
      .addPass(mlir::createInterchangeGenericOpsPass)
      .addPass(mlir::createCollapseDimsPass)
      .addPass(memref::createResolveShapedTypeResultDimsPass)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)
      /* Elementwise fusion on-tensor level,
         why not use mlir::createLinalgElementwiseOpFusionPass,
         it can't fusion when multi-use */
      .addPass([]() { return mlir::createFusionOfTensorOpsPass(); })
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)
      .addPass(mlir::createSCCPPass);
  pm.addPass(mlir::createSubgraphSplitPass(dynamic_mode));
  // pm.addPass(mlir::createRemoveDeadValuesPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // ToDo
}

} // namespace mlir

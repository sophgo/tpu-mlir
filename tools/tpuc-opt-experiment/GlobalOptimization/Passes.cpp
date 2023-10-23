#include "Passes.h"
namespace mlir {

using FunctionLikeNest = MultiOpNest<mlir::func::FuncOp>;
void buildGlobalOptimizationPassPipeline(
    OpPassManager &pm) {
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
  pm.addPass(mlir::createInlinerPass());
  //ToDo
}

}

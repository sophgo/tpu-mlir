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

  pm.addPass(mlir::createInlinerPass());
  //ToDo
}

}

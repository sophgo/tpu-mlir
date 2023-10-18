#include "include/Utils.h"
#include "Passes.h"
#include "Utils/PassUtils.h"
namespace mlir {

using FunctionLikeNest = MultiOpNest<mlir::func::FuncOp>;
void buildGlobalOptimizationPassPipeline(
    OpPassManager &pm) {
  // Preprocessing passes to get the program into a canonical state.
  FunctionLikeNest(pm)
      .addPass(mlir::createDetachElementwiseFromNamedOpsPass)
      .addPass(mlir::createCSEPass);
  pm.addPass(mlir::createInlinerPass());
  //ToDo
}
}

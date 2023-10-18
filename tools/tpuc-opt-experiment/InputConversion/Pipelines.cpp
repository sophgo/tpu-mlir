

#include "include/Utils.h"
#include "InputConversion/Passes.h"
#include "GlobalOptimization/Passes.h"
namespace mlir {

void buildPrecompileTransformPassPipeline(OpPassManager &pm, std::string target) {
    pm.addPass(mlir::createAssignTargetDevicePass(target));
    pm.addPass(mlir::createSetEntryPointPass());
    pm.addPass(createAutoInputConversionPipelinePass());

    buildGlobalOptimizationPassPipeline(pm);

    /*pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
    pm.addPass(mlir::createInlinerPass());*/
}

}

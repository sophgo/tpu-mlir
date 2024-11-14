#pragma once
#include "include/Utils.h"
namespace mlir {

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createAssignTargetDevicePass(std::string targets);

std::unique_ptr<OperationPass<ModuleOp>> createSetEntryPointPass();

std::unique_ptr<OperationPass<ModuleOp>>
createAutoInputConversionPipelinePass();
#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "InputConversion/Passes.h.inc"

} // namespace mlir

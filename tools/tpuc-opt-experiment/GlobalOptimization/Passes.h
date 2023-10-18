#pragma once

namespace  mlir
{
void buildGlobalOptimizationPassPipeline(
    OpPassManager &mainPassManager);

std::unique_ptr<Pass> createDetachElementwiseFromNamedOpsPass();

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "GlobalOptimization/Passes.h.inc"
} // namespace  mlir


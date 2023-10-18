#pragma once
#include "Utils.h"
namespace  mlir
{
void buildPrecompileTransformPassPipeline(
    OpPassManager &pm, std::string target);
} // namespace  mlir


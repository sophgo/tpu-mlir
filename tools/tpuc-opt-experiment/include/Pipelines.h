#pragma once
#include "Utils.h"
namespace mlir {
void buildPrecompileTransformPassPipeline(OpPassManager &pm, std::string target,
                                          bool dynamic_mode);
} // namespace  mlir

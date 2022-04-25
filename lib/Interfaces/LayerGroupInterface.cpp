#include "sophgo/Interfaces/LayerGroupInterface.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"

using namespace mlir;

namespace sophgo {

constexpr llvm::StringRef LayerGroupInterface::kLayerGroupAttrName;

}

#include "sophgo/Interfaces/LayerGroupInterface.cpp.inc"

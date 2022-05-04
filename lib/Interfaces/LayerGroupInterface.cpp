#include "sophgo/Interfaces/LayerGroupInterface.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"

using namespace mlir;

namespace sophgo {

constexpr llvm::StringRef LayerGroupInterface::kLayerGroupAttrName;
void LayerGroupInterface::fixSlice(int64_t &in_idx, int64_t &in_slice, int64_t in_length) {
  // avoid leak
  auto end_idx = in_idx + in_slice;
  if (in_idx < 0) {
    in_idx = 0;
  }
  if (end_idx > in_length) {
    end_idx = in_length;
  }
  in_slice = end_idx - in_idx;
}
}

#include "sophgo/Interfaces/LayerGroupInterface.cpp.inc"

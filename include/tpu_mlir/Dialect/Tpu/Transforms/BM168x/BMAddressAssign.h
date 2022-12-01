//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/GmemAllocator.hpp"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace llvm;
using namespace mlir;
namespace tpu_mlir {
namespace tpu {

class BMAddressAssign {
public:
  BMAddressAssign() {}
  void assign(mlir::ModuleOp &module);

protected:
  void updateLiveRangeofBMOps(Operation *op, int index,
                              std::map<Operation *, uint32_t> &ops_loc,
                              std::map<ValueInfo, TensorLive> &liveRange,
                              std::vector<ValueInfo> &common_ops,
                              std::vector<ValueInfo> &inplace_ops,
                              int alignment);
  void findInPlaceOpMaxUsePosition(Operation *op, uint32_t &maxPosition,
                                   std::map<Operation *, uint32_t> &ops_loc);

  bool isInPlaceOp(Operation *op);
  int getOutIndex(Operation *op, Value &out);
  uint32_t getTensorGmemSize(Operation *op, int index, int64_t aligment_);

protected:
  StringRef chip;
};

} // namespace tpu

namespace bm168x {
void populateGlobalBufferPatterns(RewritePatternSet *patterns);
}
} // namespace tpu_mlir

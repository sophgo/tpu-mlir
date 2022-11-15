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

namespace tpu_mlir {
namespace tpu {

class CVAddressAssign {
public:
  CVAddressAssign() {}
  void assign(mlir::ModuleOp &module);

protected:
  bool isOpBelongToIOMemoryRegion(Operation *op, int index,
                                  std::vector<Value> &outputs);
  bool isOpBelongToPrivateMemoryRegion(Operation *op, int index);
  bool isInPlaceOpBelongToPrivateMemoryRegion(Operation *op, int index);
  void updateLiveRangeOfOps(Operation *op, int index,
                            std::map<Operation *, uint32_t> &ops_loc,
                            std::map<ValueInfo, TensorLive> &liveRange,
                            std::vector<ValueInfo> &inplace_ops,
                            int64_t alignment = 64);

  void updateLiveRangeOfInPlaceOp(Operation *op, int i,
                                  std::map<ValueInfo, TensorLive> &liveRange,
                                  int64_t start, int64_t end,
                                  uint32_t tensor_size);

  void updateAddressOfInPlaceOp(ValueInfo &v_info);

  bool isInPlaceOp(Operation *op);

  bool isOutput(Operation *op, int index);

  void findInPlaceOpMaxUsePosition(Operation *op, uint32_t &maxPosition,
                                   std::map<Operation *, uint32_t> &ops_loc);
  int getOutIndex(Operation *op, Value &out);

  uint32_t getTensorGmemSize(Operation *op, int index, int64_t aligment_);
};
} // namespace tpu
} // namespace tpu_mlir

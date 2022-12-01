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

#include <map>
#include <vector>
namespace tpu_mlir {
namespace tpu {

enum MemType { MEM_IOMEM = 0, MEM_PRIVATE = 1, MEM_SHARED = 2 };

struct OpElement {
  OpElement() {
    live.start = 0xFFFFFFFF;
    live.end = 0;
    live.tensor_size = 0;
    mem_type = MEM_SHARED;
    need_alloc = true;
    inplace = false;
  }
  TensorLive live;
  MemType mem_type;
  ValueInfo target_v;
  bool need_alloc;
  bool inplace;
};

class CVAddressAssign {
public:
  CVAddressAssign() {}
  void assign(mlir::ModuleOp &module);

protected:
  bool isOpBelongToIOMemoryRegion(Operation *op, int index,
                                  std::vector<Value> &outputs);

  bool isOpBelongToPrivateMemoryRegion(Operation *op, int index);

  bool isInPlaceOpBelongToPrivateMemoryRegion(Operation *op, int index);

  void updateLiveRangeofPreOp(std::map<ValueInfo, OpElement> &op_infos,
                              Operation *op, uint32_t end,
                              std::map<Operation *, uint32_t> &ops_loc,
                              MemType mem_type, int64_t alignment);

  void updateLiveRangeOfInPlaceOp(std::map<ValueInfo, OpElement> &op_infos,
                                  Operation *op, uint32_t end,
                                  std::map<Operation *, uint32_t> &ops_loc,
                                  MemType mem_type, int64_t alignment);

  void updateLiveRange(Operation *op, std::map<Operation *, uint32_t> &ops_loc,
                       std::map<ValueInfo, OpElement> &op_infos,
                       std::vector<mlir::Value> &outputs, int64_t alignment);

  void updateAddressOfInPlaceOp(ValueInfo &v_info,
                                std::map<ValueInfo, OpElement> &op_infos,
                                int64_t alignment);

  bool isInPlaceOp(Operation *op);

  bool isOutput(Operation *op, int index);

  void findInPlaceOpMaxUsePosition(Operation *op, uint32_t &maxPosition,
                                   std::map<Operation *, uint32_t> &ops_loc);

  int getOutIndex(Operation *op, Value &out);

  uint32_t getTensorGmemSize(Operation *op, int index, int64_t aligment_);
};
} // namespace tpu
} // namespace tpu_mlir

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/GmemAllocator.h"
#include "tpu_mlir/Support/Module.h"

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
  void assign(mlir::ModuleOp &module, bool reuse_addr, bool merge_weight,
              bool compress_weight, std::string &weight_map_file);

protected:
  std::string calcMD5(std::vector<uint8_t> &data);

  bool loadAddressMapping(
      std::string &mapFileName,
      std::unordered_map<std::string, std::pair<int64_t, int64_t>>
          &addrMapping);

  void checkIfFileGood(std::string &fileName,
                       std::unique_ptr<std::fstream> &stream);

  void assign_weight_addr(mlir::ModuleOp &module, bool merge_weight,
                          bool compress_weight, std::string &weight_map_file);

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
                       std::vector<ValueInfo> &inplace_ops,
                       std::vector<mlir::Value> &outputs, int64_t alignment);

  void updateAddressOfInPlaceOp(ValueInfo &v_info,
                                std::map<ValueInfo, OpElement> &op_infos,
                                int64_t alignment);

  bool isInPlaceOp(Operation *op);

  bool isOutput(Operation *op, int index);

  uint32_t getTensorGmemSize(Operation *op, int index, int64_t aligment_);

  void updateConcatOpTargetV(std::vector<ValueInfo> &inplace_ops,
                             std::map<ValueInfo, OpElement> &op_infos);
};
} // namespace tpu
} // namespace tpu_mlir

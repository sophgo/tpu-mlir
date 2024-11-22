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

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

class BMAddressAssign {
public:
  BMAddressAssign() {}
  void assign(ModuleOp &module, bool reuse_addr);
  static bool isInPlaceOp(Operation *op);

protected:
  void updateLiveRangeofBMOps(Operation *op, int index,
                              std::map<Operation *, uint32_t> &ops_loc,
                              std::map<ValueInfo, TensorLive> &liveRange,
                              std::vector<ValueInfo> &common_ops,
                              std::vector<ValueInfo> &inplace_ops,
                              int alignment);
  void findInPlaceOpMaxUsePosition(Operation *op, uint32_t &maxPosition,
                                   std::map<Operation *, uint32_t> &ops_loc);
  int getOutIndex(Operation *op, Value &out);
  uint32_t getTensorGmemSize(Operation *op, int index, int64_t aligment_);
  bool is_next_subnet_input(Operation *op, int index);
  void updateAddressByAddrMode(mlir::ModuleOp &m, int64_t start_addr,
                               int64_t addr_limit);
  std::vector<uint32_t>
  getConcatOpLive(Operation *op, std::map<ValueInfo, TensorLive> &liveRange);
  void assignL2SRAM(ModuleOp &module);
  void assignAfter(ModuleOp &module, std::vector<ValueInfo> &inplace_ops);

protected:
  StringRef chip;
};

} // namespace tpu
} // namespace tpu_mlir

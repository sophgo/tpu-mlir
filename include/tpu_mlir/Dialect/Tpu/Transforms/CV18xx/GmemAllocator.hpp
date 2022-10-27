///===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//


#pragma once
#include <list>
#include <map>
#include <set>

#include "mlir/Support/LLVM.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"

#include "GmemAllocatorMethod.h"

namespace tpu_mlir {
namespace tpu {

class GmemAllocator {
public:
  GmemAllocator(
      std::map<int64_t, int64_t> &gaddrMap,
      uint32_t alignment = 16);
  void registerMethod(std::string method_name, bool reuse);
  void registerAllMethod();
  int64_t assignGaddr(
      std::vector<int64_t> &ops,
      std::map<int64_t, TensorLive> &liveRange,
      bool neuronMemoryReuse, int64_t baseGaddr);
  static void markGmemReusedOp(
      std::vector<int64_t> &ops,
      std::map<int64_t, int64_t> &gaddrMap,
      std::map<int64_t, TensorLive> &liveRange,
      std::set<int64_t> &gmemReusedSet,
      uint32_t alignment);
  static int64_t assignSpecifiedGmemToOp(
      Operation *op,
      std::map<int64_t, int64_t> &gaddrMap,
      int64_t baseGaddr,
      uint32_t alignment);

  std::map<int64_t, int64_t> &gaddrMap_;
  uint32_t alignment;
private:
  std::vector<std::string> reuse_methods_;
  std::vector<std::string> methods_;
};

} // namespace tpu
} // namespcae tpu_mlir

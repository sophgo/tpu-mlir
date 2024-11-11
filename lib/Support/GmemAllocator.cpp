//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GmemAllocator.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

GmemAllocator::GmemAllocator(std::map<ValueInfo, int64_t> &gaddrMap,
                             uint32_t alignment)
    : gaddrMap_(gaddrMap), alignment(alignment) {}

void GmemAllocator::markGmemReusedOp(std::vector<ValueInfo> &ops,
                                     std::map<ValueInfo, int64_t> &gaddrMap,
                                     std::map<ValueInfo, TensorLive> &liveRange,
                                     std::set<ValueInfo> &gmemReusedSet,
                                     uint32_t alignment) {

  std::vector<ValueInfo> tmp;
  for (int i = ops.size() - 1; i >= 0; i--) {
    if (gaddrMap.find(ops[i]) == gaddrMap.end())
      continue;
    auto addr_i = gaddrMap[ops[i]];
    auto sz_i = liveRange[ops[i]].tensor_size;
    for (int j = 0; j < (int)tmp.size(); j++) {
      auto addr_j = gaddrMap[tmp[j]];
      auto sz_j = liveRange[tmp[j]].tensor_size;
      auto start = std::min(addr_i, addr_j);
      auto end = std::max(addr_i + sz_i, addr_j + sz_j);
      // memory overlap
      if (end - start < sz_i + sz_j) {
        gmemReusedSet.insert(ops[i]);
      }
    }
    tmp.push_back(ops[i]);
  }
}

void GmemAllocator::registerMethod(std::string method_name, bool reuse) {
  if (reuse) {
    reuse_methods_.emplace_back(method_name);
  } else {
    methods_.emplace_back(method_name);
  }
}

void GmemAllocator::registerAllMethod() {
  registerMethod("FitFirstAssign", true);
  registerMethod("FitFirstAssign", false);
  registerMethod("OpSizeOrderAssign", true);
}

int64_t GmemAllocator::assignGaddr(std::vector<ValueInfo> &ops,
                                   std::map<ValueInfo, TensorLive> &liveRange,
                                   bool neuronMemoryReuse, int64_t baseGaddr) {
  if (ops.empty()) {
    LAYER_GROUP_LOG_DEBUG_BLOCK(
        { llvm::outs() << "Warning input ops is empty!\n"; });
    return 0;
  }
  // for special op, remove ops which not in liveRange, addr is assigned later
  for (auto it = ops.begin(); it != ops.end();) {
    if (liveRange.find(*it) == liveRange.end()) {
      it = ops.erase(it);
      continue;
    }
    ++it;
  }

  if (!reuse_methods_.size() && !methods_.size()) {
    registerAllMethod();
  }

  std::vector<std::string> *cur_methods;
  if (neuronMemoryReuse) {
    cur_methods = &reuse_methods_;
  } else {
    cur_methods = &methods_;
  }

  std::vector<std::unique_ptr<GmemAllocatorMethod>> alloc_methods;
  for (auto &name : *cur_methods) {
    auto p = GmemAllocatorMethodFactory::makeMethod(name, gaddrMap_, alignment);
    if (p) {
      alloc_methods.emplace_back(p);
    } else {
      assert(0);
    }
  }

  int64_t min_gmem_size = 0;
  int idx = 0;
  for (uint32_t i = 0; i < alloc_methods.size(); ++i) {
    int64_t gmem_size = alloc_methods[i]->assignGaddr(
        ops, liveRange, neuronMemoryReuse, baseGaddr);
    if (gmem_size < min_gmem_size || min_gmem_size == 0) {
      min_gmem_size = gmem_size;
      idx = i;
    }
  }
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::outs() << "GmemAllocator use " << alloc_methods[idx]->getName()
                 << "\n";
  });
  auto addrMap = alloc_methods[idx]->getAddrMap();
  gaddrMap_.swap(addrMap);
  return min_gmem_size;
}

int part(std::vector<ValueInfo> &ops, int low, int high,
         std::map<ValueInfo, TensorLive> &liveRange) {
  int i = low, j = high;
  uint32_t pivot = liveRange[ops[low]].start;
  while (i < j) {
    while (i < j && liveRange[ops[j]].start > pivot) {
      j--;
    }
    if (i < j) {
      std::swap(ops[i++], ops[j]);
    }
    while (i < j && liveRange[ops[i]].start <= pivot) {
      i++;
    }
    if (i < j) {
      std::swap(ops[i], ops[j--]);
    }
  }
  return i;
}

void quickSort(std::vector<ValueInfo> &ops, int low, int high,
               std::map<ValueInfo, TensorLive> &liveRange) {
  int mid;
  if (low < high) {
    mid = part(ops, low, high, liveRange);
    quickSort(ops, low, mid - 1, liveRange);
    quickSort(ops, mid + 1, high, liveRange);
  }
}

void GmemAllocator::sortOpByLiveStart(
    std::vector<ValueInfo> &ops, std::map<ValueInfo, TensorLive> &liveRange) {
  // use quicksort algorithm
  quickSort(ops, 0, ops.size() - 1, liveRange);
}
} // namespace tpu
} // namespace tpu_mlir

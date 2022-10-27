//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/GmemAllocator.hpp"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <sstream>
#include <fstream>
#include <set>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {

GmemAllocator::GmemAllocator(
    std::map<int64_t, int64_t> &gaddrMap,
    uint32_t alignment)
    : gaddrMap_(gaddrMap),
      alignment(alignment) {
}

void GmemAllocator::markGmemReusedOp(
      std::vector<int64_t> &ops,
      std::map<int64_t, int64_t> &gaddrMap,
      std::map<int64_t, TensorLive> &liveRange,
      std::set<int64_t> &gmemReusedSet,
      uint32_t alignment) {

  std::vector<int64_t> tmp;
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

int64_t GmemAllocator::assignSpecifiedGmemToOp(
                                       Operation *op,
                                       std::map<int64_t, int64_t> &gaddrMap,
                                       int64_t baseGaddr,
                                       uint32_t alignment) {
                                         /*
  int64_t size = 0;
  if (auto concatOp = dyn_cast_or_null<tpu::ConcatNOp>(op)) {
    size = GmemAllocatorMethod::getTensorGmemSize(op, alignment);
    gaddrMap[op] = baseGaddr;
  }
  return size;
  */
  return 0;
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

int64_t GmemAllocator::assignGaddr(
    std::vector<int64_t> &ops,
    std::map<int64_t, TensorLive> &liveRange,
    bool neuronMemoryReuse, int64_t baseGaddr) {
  if (ops.empty()) {
    llvm::errs() << "Warning input ops is empty!\n";
    return 0;
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

  std::vector<std::unique_ptr<GmemAllocatorMethod> > alloc_methods;
  for (auto& name : *cur_methods) {
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
    int64_t gmem_size = alloc_methods[i]->assignGaddr(ops, liveRange, neuronMemoryReuse, baseGaddr);
    if (gmem_size < min_gmem_size || min_gmem_size == 0) {
      min_gmem_size = gmem_size;
      idx = i;
    }
  }

  llvm::errs() << "GmemAllocator use " << alloc_methods[idx]->getName() << "\n";
  gaddrMap_.swap(alloc_methods[idx]->gaddrMap_);
  return min_gmem_size;
}
} // tpu
} //tpu_mlir

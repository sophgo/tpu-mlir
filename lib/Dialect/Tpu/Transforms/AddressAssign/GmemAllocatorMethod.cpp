//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "GmemAllocatorMethod.h"
#include <limits>
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "gmem-allocator"
using namespace tpu_mlir::tpu;

namespace tpu_mlir {
namespace tpu {

GmemAllocatorMethod::GmemAllocatorMethod(std::map<ValueInfo, int64_t> &gaddrMap,
                                         uint32_t aligment)
    : gaddrMap_(gaddrMap), aligment_(aligment) {
  GmemBlock block;
  block.start = 0;
  block.size = 0xFFFFFFFF;
  block.v_info = ValueInfo(0, -1);
  std::list<GmemBlock> snapshot;
  snapshot.emplace_back(block);
  album_.emplace_back(snapshot);
}
GmemAllocatorMethod::~GmemAllocatorMethod() {}

std::string GmemAllocatorMethod::getName() { return name_; }

void GmemAllocatorMethod::reuseGmemBlock(
    std::list<GmemBlock> &snapshot, ValueInfo &v,
    std::map<ValueInfo, TensorLive> &liveRange) {
  // int tensor_idx = findValueRange(liveRange, tensor);
  for (auto &blk : snapshot) {
    if (!blk.v_info.op) {
      continue;
    }
    // free the block if end position of block's op
    // is same as current op's start position
    if (liveRange[blk.v_info].end <= liveRange[v].start ||
        liveRange[blk.v_info].start >= liveRange[v].end) {
      blk.v_info.op = 0;
    }
  }
  // merge contiguous free blocks into one
  mergeFreeGmemBlocks(snapshot);
}

int64_t GmemAllocatorMethod::allocGmemBlock(std::list<GmemBlock> &snapshot,
                                            ValueInfo &v,
                                            uint32_t tensor_size) {
  auto last = --snapshot.end();
  auto selected = last;

  // Policy: just select the free block that has largest size.
  // TODO, we can try other policy here.
  uint32_t max_free_size = 0;
  for (auto iter = snapshot.begin(); iter != last; ++iter) {
    if (!iter->v_info.op && iter->size > max_free_size) {
      selected = iter;
      max_free_size = iter->size;
    }
  }
  gaddrMap_[v] = -1;

  auto gsize = tensor_size;
  auto s_addr = selected->start;

  if (selected->size > gsize) {
    // Occupy this free block firstly.
    // Split the remain memory to anther block,
    // and insert it into snapshot.
    GmemBlock blk;
    blk.start = selected->start + gsize;
    blk.size = selected->size - gsize;
    blk.v_info.op = 0;

    selected->v_info = v;
    selected->size = gsize;
    snapshot.insert(++selected, blk);
  } else {
    selected->v_info = v;
    selected->size = gsize;

    // Enlarge the block to match the size of tensor,
    // and correct the offset of subsequent blocks.
    int64_t offset = selected->start + selected->size;
    while (++selected != snapshot.end()) {
      selected->start = offset;
      offset += selected->size;
    }
  }
  return s_addr;
}

void GmemAllocatorMethod::mergeFreeGmemBlocks(std::list<GmemBlock> &snapshot) {
  auto iter = snapshot.begin();
  while (iter != snapshot.end()) {
    auto cur = iter++;
    while (iter != snapshot.end() && !cur->v_info.op && !iter->v_info.op) {
      cur->size += iter->size;
      snapshot.erase(iter++);
    }
  }
}

void GmemAllocatorMethod::backPropagateToAssignGaddr() {
  for (int i = album_.size() - 1; i >= 0; --i) {
    auto &snapshot = album_[i];
    int64_t offset = 0;
    for (auto &blk : snapshot) {
      if (!blk.v_info.op) {
        blk.start = offset;
        offset += blk.size;
        continue;
      }
      auto v = blk.v_info;
      // if tensor was allocated, relocate the start offset of block.
      blk.start = (gaddrMap_[v] == -1) ? blk.start : gaddrMap_[v];
      // if start offset of block is already allocated,
      // relocate it to current offset point.
      if (blk.start <= offset) {
        blk.start = offset;
      }
      offset = blk.start + blk.size;

      if (gaddrMap_[v] == -1) {
        gaddrMap_[v] = blk.start;
      }
    }
  }
}

int64_t GmemAllocatorMethod::updateGmemUsedStatistic(
    std::vector<ValueInfo> &ops, std::map<ValueInfo, TensorLive> &liveRange) {
  int64_t totalNeuronSize = 0;
  int64_t totalGmemUsed = 0;
  for (int i = ops.size() - 1; i >= 0; i--) {
    auto addr_i = gaddrMap_[ops[i]];
    auto sz_i = liveRange[ops[i]].tensor_size;
    if (totalGmemUsed < addr_i + sz_i) {
      totalGmemUsed = addr_i + sz_i;
    }
    totalNeuronSize += sz_i;
  }

  int32_t reuseRate = 0;
  if (totalNeuronSize) {
    reuseRate =
        (int32_t)((totalNeuronSize - totalGmemUsed) * 100 / totalNeuronSize);
  }

  llvm::errs() << "GmemAllocMethod:" << name_.c_str()
               << "  Gmem Used: " << totalGmemUsed << "/" << totalNeuronSize
               << ", gmem reused rate:" << reuseRate << "%\n";
  return totalGmemUsed;
}

GmemAllocFitFirst::GmemAllocFitFirst(std::map<ValueInfo, int64_t> &gaddrMap,
                                     uint32_t aligment)
    : GmemAllocatorMethod(gaddrMap, aligment) {
  name_ = "FitFirstAssign";
}

int64_t
GmemAllocFitFirst::assignGaddr(std::vector<ValueInfo> &ops,
                               std::map<ValueInfo, TensorLive> &liveRange,
                               bool neuronMemoryReuse, int64_t baseGaddr) {
  for (auto op : ops) {
    // llvm::errs() << "loop #" << album_.size() - 1 << "\n";
    auto snapshot = album_[album_.size() - 1];
    if (neuronMemoryReuse) {
      reuseGmemBlock(snapshot, op, liveRange);
    }
    allocGmemBlock(snapshot, op, liveRange[op].tensor_size);
    album_.push_back(snapshot);
  }

  {
#if 0
  int i = 0;
  for (auto snapshot : album_) {
    llvm::errs() << "Snapshot idx:" << i++ << "\n";
    int j = 0;
    for (auto &blk : snapshot) {
      llvm::errs() << "\t" << j++ << " "
                   << (blk.op ? blk.op->getName().getStringRef()
                              : llvm::StringRef("null"))
                   << ":"
                   << (blk.op ? module::getName(blk.op) : llvm::StringRef("null"))
                   << ", start:" << blk.start << ", size:" << blk.size
                   << ", free:" << (blk.op ? false : true) << "\n";
    }
  }
#endif
  }

  backPropagateToAssignGaddr();
  auto totalGmemUsed = updateGmemUsedStatistic(ops, liveRange);
  // update gaddr map by adding base gaddr.
  for (auto op : ops) {
    gaddrMap_[op] += baseGaddr;
  }

  for (auto op : ops) {
    auto out_index = liveRange[op].out_index;
    auto tensor_size = liveRange[op].tensor_size;
    auto real_op = (Operation *)(op.op);
    llvm::errs() << "op:" << real_op->getName()
                 << ", name:" << module::getName(real_op->getResult(out_index))
                 << ", addr:" << gaddrMap_[op] << ", baseGaddr:" << baseGaddr
                 << ", size:" << tensor_size
                 << ", end:" << gaddrMap_[op] + tensor_size
                 << ", range:" << liveRange[op].start << " ~ "
                 << liveRange[op].end << "\n";
  }
  return totalGmemUsed;
}

GmemAllocOpSizeOrder::GmemAllocOpSizeOrder(
    std::map<ValueInfo, int64_t> &gaddrMap, uint32_t aligment)
    : GmemAllocatorMethod(gaddrMap, aligment) {
  name_ = "OpSizeOrderAssign";
}

int64_t
GmemAllocOpSizeOrder::assignGaddr(std::vector<ValueInfo> &ops,
                                  std::map<ValueInfo, TensorLive> &liveRange,
                                  bool neuronMemoryReuse, int64_t baseGaddr) {

  std::list<std::shared_ptr<OpAddr>> op_list;
  std::list<std::shared_ptr<OpAddr>> allocated_op_list;
  assert(neuronMemoryReuse);
  for (auto op : ops) {
    // int addr_idx = findValueAddr(gaddrMap_, tensor);
    uint32_t op_size = liveRange[op].tensor_size;
    std::shared_ptr<OpAddr> op_addr = std::make_shared<OpAddr>(
        op, op_size, liveRange[op].start, liveRange[op].end);
    op_list.emplace_back(op_addr);
    gaddrMap_[op] = -1;
  }

  op_list.sort([](std::shared_ptr<OpAddr> &a, std::shared_ptr<OpAddr> &b) {
    return a->size >= b->size;
  });

  int64_t total_consumption = 0;
  for (auto &op_addr : op_list) {
    int64_t prev_offset = 0;
    int64_t best_offset = -1;
    int64_t smallest_gap = std::numeric_limits<int64_t>::max();
    for (auto &allocated_op_addr : allocated_op_list) {
      uint32_t max_first_pos =
          std::max(op_addr->first_pos, allocated_op_addr->first_pos);
      uint32_t min_last_pos =
          std::min(op_addr->end_pos, allocated_op_addr->end_pos);
      if (max_first_pos < min_last_pos) {
        int64_t gap = allocated_op_addr->start - prev_offset;
        if (gap >= op_addr->size && gap < smallest_gap) {
          smallest_gap = gap;
          best_offset = prev_offset;
        }
        prev_offset = std::max(prev_offset, allocated_op_addr->end);
      }
    }
    if (best_offset == -1) {
      best_offset = prev_offset;
    }
    op_addr->start = best_offset;
    op_addr->end = op_addr->start + op_addr->size;
    total_consumption = std::max(total_consumption, op_addr->end);
    auto iter = std::find_if(allocated_op_list.begin(), allocated_op_list.end(),
                             [&op_addr](std::shared_ptr<OpAddr> &p) {
                               return p->start >= op_addr->start;
                             });
    allocated_op_list.emplace(iter, op_addr);
  }

  int64_t totalNeuronSize = 0;
  for (auto &op_addr : allocated_op_list) {
    if (gaddrMap_[op_addr->op] == -1) {
      gaddrMap_[op_addr->op] = op_addr->start;
    }
    totalNeuronSize += op_addr->size;
  }

  int32_t reuseRate = 0;
  if (totalNeuronSize) {
    reuseRate = (int32_t)((totalNeuronSize - total_consumption) * 100 /
                          totalNeuronSize);
  }

  llvm::errs() << "GmemAllocMethod:" << name_.c_str()
               << "  Gmem Used: " << total_consumption << "/" << totalNeuronSize
               << ", gmem reused rate:" << reuseRate << "%\n";

  for (auto &op_addr : allocated_op_list) {
    // update gaddr map by adding base gaddr.
    gaddrMap_[op_addr->op] += baseGaddr;
  }

  for (auto op : ops) {
    auto out_index = liveRange[op].out_index;
    auto tensor_size = liveRange[op].tensor_size;
    auto real_op = (Operation *)(op.op);
    llvm::errs() << "op:" << real_op->getName()
                 << ", name:" << module::getName(real_op->getResult(out_index))
                 << ", addr:" << gaddrMap_[op] << ", baseGaddr:" << baseGaddr
                 << ", size:" << tensor_size
                 << ", end:" << gaddrMap_[op] + tensor_size
                 << ", range:" << liveRange[op].start << " ~ "
                 << liveRange[op].end << "\n";
  }
  return total_consumption;
}
} // namespace tpu
} // namespace tpu_mlir

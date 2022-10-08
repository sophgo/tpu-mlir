#include <limits>
#include <llvm/Support/Debug.h>
#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/GmemAllocatorMethod.h"

#define DEBUG_TYPE "gmem-allocator"

namespace tpu_mlir {
namespace tpu {


uint32_t GmemAllocatorMethod::getTensorGmemSize(Operation *op, uint32_t aligment_) {
  uint32_t size = Module::getBytes(op->getResult(0));
  // pad to aligment_
  if (size % aligment_) {
    size = size + aligment_ - (size % aligment_);
  }
  return size;
}

GmemAllocatorMethod::GmemAllocatorMethod(
    std::map<Operation *, int64_t> &gaddrMap, uint32_t aligment)
    : gaddrMap_(gaddrMap), aligment_(aligment) {
  GmemBlock block;
  block.start = 0;
  block.size = 0xFFFFFFFF;
  block.op = nullptr;
  std::list<GmemBlock> snapshot;
  snapshot.emplace_back(block);
  album_.emplace_back(snapshot);
}
GmemAllocatorMethod::~GmemAllocatorMethod() {}

std::string GmemAllocatorMethod::getName() { return name_; }

void GmemAllocatorMethod::reuseGmemBlock(
    std::list<GmemBlock> &snapshot, Operation *op,
    std::map<Operation *, std::vector<uint32_t>> &liveRange) {
  for (auto &blk : snapshot) {
    if (!blk.op) {
      continue;
    }
    // free the block if end position of block's op
    // is same as current op's start position
    if (liveRange[blk.op][1] <= liveRange[op][0] ||
        liveRange[blk.op][0] >= liveRange[op][1]) {
      blk.op = nullptr;
    }
  }
  // merge contiguous free blocks into one
  mergeFreeGmemBlocks(snapshot);
}

int64_t
GmemAllocatorMethod::allocGmemBlock(std::list<GmemBlock> &snapshot,
                                    Operation *op) {
  auto last = --snapshot.end();
  auto selected = last;

  // Policy: just select the free block that has largest size.
  // TODO, we can try other policy here.
  uint32_t max_free_size = 0;
  for (auto iter = snapshot.begin(); iter != last; ++iter) {
    if (!iter->op && iter->size > max_free_size) {
      selected = iter;
      max_free_size = iter->size;
    }
  }

  gaddrMap_[op] = -1;
  auto gsize = getTensorGmemSize(op, aligment_);
  auto s_addr = selected->start;

  if (selected->size > gsize) {
    // Occupy this free block firstly.
    // Split the remain memory to anther block,
    // and insert it into snapshot.
    GmemBlock blk;
    blk.start = selected->start + gsize;
    blk.size = selected->size - gsize;
    blk.op = nullptr;

    selected->op = op;
    selected->size = gsize;
    snapshot.insert(++selected, blk);
  } else {
    selected->op = op;
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
    while (iter != snapshot.end() && !cur->op && !iter->op) {
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
      if (!blk.op) {
        blk.start = offset;
        offset += blk.size;
        continue;
      }
      auto op = blk.op;
      // if tensor was allocated, relocate the start offset of block.
      blk.start = (gaddrMap_[op] == -1) ? blk.start : gaddrMap_[op];
      // if start offset of block is already allocated,
      // relocate it to current offset point.
      if (blk.start <= offset) {
        blk.start = offset;
      }
      offset = blk.start + blk.size;

      if (gaddrMap_[op] == -1) {
        gaddrMap_[op] = blk.start;
      }
    }
  }
}

int64_t GmemAllocatorMethod::updateGmemUsedStatistic(std::vector<Operation *> &ops) {
  int64_t totalNeuronSize = 0;
  int64_t totalGmemUsed = 0;
  for (int i = ops.size() - 1; i >= 0; i--) {
    auto addr_i = gaddrMap_[ops[i]];
    auto sz_i = getTensorGmemSize(ops[i], aligment_);
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

  llvm::errs() << "GmemAllocMethod:" << name_.c_str() << "  Gmem Used: " << totalGmemUsed << "/" << totalNeuronSize
               << ", gmem reused rate:" << reuseRate << "%\n";
  return totalGmemUsed;
}

GmemAllocFitFirst::GmemAllocFitFirst(std::map<Operation *, int64_t> &gaddrMap, uint32_t aligment)
    : GmemAllocatorMethod(gaddrMap, aligment) {
  name_ = "FitFirstAssign";
}

int64_t GmemAllocFitFirst::assignGaddr(std::vector<Operation *> &ops,
                    std::map<Operation *, std::vector<uint32_t>> &liveRange,
                    bool neuronMemoryReuse, int64_t baseGaddr) {
  for (auto op : ops) {
    // llvm::errs() << "loop #" << album_.size() - 1 << "\n";
    auto snapshot = album_[album_.size() - 1];
    if (neuronMemoryReuse) {
      reuseGmemBlock(snapshot, op, liveRange);
    }
    allocGmemBlock(snapshot, op);
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
                   << (blk.op ? Module::getName(blk.op) : llvm::StringRef("null"))
                   << ", start:" << blk.start << ", size:" << blk.size
                   << ", free:" << (blk.op ? false : true) << "\n";
    }
  }
  #endif
}

  backPropagateToAssignGaddr();
  auto totalGmemUsed = updateGmemUsedStatistic(ops);
  // update gaddr map by adding base gaddr.
  for (auto op : ops) {
    gaddrMap_[op] += baseGaddr;
  }

  for (auto op : ops) {
    llvm::errs() << "op:" << op->getName() << ", name:" << Module::getName(op)
                 << ", addr:" << gaddrMap_[op] << ", baseGaddr:" << baseGaddr
                 << ", size:" << getTensorGmemSize(op, aligment_)
                 << ", end:" << gaddrMap_[op] + getTensorGmemSize(op, aligment_)
                 << ", range:" << liveRange[op][0] << " ~ " << liveRange[op][1]
                 << "\n";
  }
  return totalGmemUsed;
}

GmemAllocOpSizeOrder::GmemAllocOpSizeOrder(std::map<Operation *, int64_t> &gaddrMap, uint32_t aligment)
    : GmemAllocatorMethod(gaddrMap, aligment) {
  name_ = "OpSizeOrderAssign";
}

int64_t GmemAllocOpSizeOrder::assignGaddr(std::vector<Operation *> &ops,
                    std::map<Operation *, std::vector<uint32_t>> &liveRange,
                    bool neuronMemoryReuse, int64_t baseGaddr) {

  std::list<std::shared_ptr<OpAddr>> op_list;
  std::list<std::shared_ptr<OpAddr>> allocated_op_list;
  assert(neuronMemoryReuse);
  for (auto &op : ops) {
    uint32_t op_size = getTensorGmemSize(op, aligment_);
    std::shared_ptr<OpAddr> op_addr = std::make_shared<OpAddr>(
        op, op_size, liveRange[op][0], liveRange[op][1]);
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
      uint32_t max_first_pos = std::max(op_addr->first_pos, allocated_op_addr->first_pos);
      uint32_t min_last_pos = std::min(op_addr->end_pos, allocated_op_addr->end_pos);
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
    auto iter = std::find_if(
        allocated_op_list.begin(), allocated_op_list.end(),
        [&op_addr](std::shared_ptr<OpAddr> &p) { return p->start >= op_addr->start; });
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
    reuseRate =
        (int32_t)((totalNeuronSize - total_consumption) * 100 / totalNeuronSize);
  }

  llvm::errs() << "GmemAllocMethod:" << name_.c_str() << "  Gmem Used: " << total_consumption << "/" << totalNeuronSize
               << ", gmem reused rate:" << reuseRate << "%\n";

  for (auto op : ops) {
    LLVM_DEBUG(llvm::errs() << "op:" << op->getName() << ", name:" << Module::getName(op)
                 << ", addr:" << gaddrMap_[op] << ", baseGaddr:" << baseGaddr
                 << ", size:" << getTensorGmemSize(op, aligment_)
                 << ", end:" << gaddrMap_[op] + getTensorGmemSize(op, aligment_)
                 << ", range:" << liveRange[op][0] << " ~ " << liveRange[op][1]
                 << "\n";);
  }

  for (auto &op_addr : allocated_op_list) {
    // update gaddr map by adding base gaddr.
    gaddrMap_[op_addr->op] += baseGaddr;
  }

  return total_consumption;
}
} //namespace tpu
} //namespace tpu_mlir

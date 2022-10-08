///===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <set>
#include <list>

#include "mlir/Support/LLVM.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"

namespace tpu_mlir {
namespace tpu {

class GmemBlock {
public:
  int64_t start;
  uint64_t size;
  Operation *op;
};

class GmemAllocatorMethod {
public:
  GmemAllocatorMethod(std::map<Operation *, int64_t> &gaddrMap, uint32_t aligment);
  virtual ~GmemAllocatorMethod();

  virtual std::string getName();

  virtual int64_t assignGaddr(
      std::vector<Operation *> &ops,
      std::map<Operation *, std::vector<uint32_t>> &liveRange,
      bool neuronMemoryReuse, int64_t baseGaddr) = 0;

  virtual void reuseGmemBlock(
      std::list<GmemBlock> &snapshot, Operation *op,
      std::map<Operation *, std::vector<uint32_t>> &liveRange);

  virtual int64_t allocGmemBlock(std::list<GmemBlock> &snapshot,
                                 Operation *op);

  virtual void mergeFreeGmemBlocks(std::list<GmemBlock> &snapshot);

  virtual void backPropagateToAssignGaddr();

  virtual int64_t updateGmemUsedStatistic(std::vector<Operation *> &ops);

  static uint32_t getTensorGmemSize(Operation *op, uint32_t aligment_);

public:
  std::map<Operation *, int64_t> gaddrMap_;

protected:
  std::string name_;
  uint32_t aligment_;
  std::vector<std::list<GmemBlock> > album_;
};

class GmemAllocFitFirst : public GmemAllocatorMethod {
public:
  GmemAllocFitFirst(std::map<Operation *, int64_t> &gaddrMap,
                    uint32_t aligment);

  int64_t assignGaddr(std::vector<Operation *> &ops,
                      std::map<Operation *, std::vector<uint32_t>> &liveRange,
                      bool neuronMemoryReuse, int64_t baseGaddr) override;
};

class GmemAllocOpSizeOrder : public GmemAllocatorMethod {
public:
  struct OpAddr {
    Operation *op;
    int64_t start = 0;
    int64_t end = 0;
    uint32_t size = 0;
    uint32_t first_pos = 0;
    uint32_t end_pos = 0;

    OpAddr(Operation *_op, uint32_t _size, uint32_t _first_pos, uint32_t _end_pos) {
      op = _op;
      size = _size;
      first_pos = _first_pos;
      end_pos = _end_pos;
    }
  };
  typedef std::list<std::shared_ptr<OpAddr>> LineSet;

public:
  GmemAllocOpSizeOrder(std::map<Operation *, int64_t> &gaddrMap,
                      uint32_t aligment);

  int64_t assignGaddr(std::vector<Operation *> &ops,
                      std::map<Operation *, std::vector<uint32_t>> &liveRange,
                      bool neuronMemoryReuse, int64_t baseGaddr) override;
};

class GmemAllocatorMethodFactory {
public:
  static GmemAllocatorMethod *
  makeMethod(std::string method_name, std::map<Operation *, int64_t> &gaddrMap,
             uint32_t aligment) {
    if (method_name == "FitFirstAssign") {
      return static_cast<GmemAllocatorMethod*>(new GmemAllocFitFirst(gaddrMap, aligment));
    } else if (method_name == "OpSizeOrderAssign") {
      return static_cast<GmemAllocatorMethod*>(new GmemAllocOpSizeOrder(gaddrMap, aligment));
    } else {
      assert(0);
      return nullptr;
    }
  }
};
} // namespace tpu
} // namespcae tpu_mlir

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
#include <set>

#include "mlir/Support/LLVM.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "Module.h"

namespace tpu_mlir {
namespace tpu {

struct ValueInfo {
  ValueInfo() {}
  ValueInfo(void *_op, int _index) : op(_op), index(_index) {}
  void *op = nullptr;
  int index = -1;
  bool operator<(const ValueInfo &rhs) const {
    if (op < rhs.op) {
      return true;
    } else if (op == rhs.op) {
      return index < rhs.index;
    } else {
      return false;
    }
  }
};

struct GmemBlock {
  GmemBlock() {};
  int64_t start;
  uint64_t size;
  ValueInfo v_info;
};

struct TensorLive {
  uint32_t start;       // start liverange
  uint32_t end;         // end liverange
  uint32_t tensor_size; // size of of output tensor
  TensorLive() {}
  TensorLive(uint32_t _start, uint32_t _end, uint32_t _tensor_size) {
    start = _start;
    end = _end;
    tensor_size = _tensor_size;
  }
};

class GmemAllocatorMethod {
public:
  GmemAllocatorMethod(std::map<ValueInfo, int64_t> &gaddrMap,
                      uint32_t aligment);
  virtual ~GmemAllocatorMethod();

  virtual std::string getName();

  virtual int64_t assignGaddr(std::vector<ValueInfo> &ops,
                              std::map<ValueInfo, TensorLive> &liveRange,
                              bool neuronMemoryReuse, int64_t baseGaddr) = 0;

  virtual void reuseGmemBlock(std::list<GmemBlock> &snapshot, ValueInfo &v,
                              std::map<ValueInfo, TensorLive> &liveRange);

  virtual int64_t allocGmemBlock(std::list<GmemBlock> &snapshot, ValueInfo &op,
                                 uint32_t tensor_size);

  virtual void mergeFreeGmemBlocks(std::list<GmemBlock> &snapshot);

  virtual void backPropagateToAssignGaddr();

  virtual int64_t
  updateGmemUsedStatistic(std::vector<ValueInfo> &ops,
                          std::map<ValueInfo, TensorLive> &liveRange);

  // static uint32_t getTensorGmemSize(Value &tensor, uint32_t aligment_);

public:
  std::map<ValueInfo, int64_t> gaddrMap_;

protected:
  std::string name_;
  uint32_t aligment_;
  std::vector<std::list<GmemBlock>> album_;
};

class GmemAllocFitFirst : public GmemAllocatorMethod {
public:
  GmemAllocFitFirst(std::map<ValueInfo, int64_t> &gaddrMap, uint32_t aligment);

  int64_t assignGaddr(std::vector<ValueInfo> &ops,
                      std::map<ValueInfo, TensorLive> &liveRange,
                      bool neuronMemoryReuse, int64_t baseGaddr) override;
};

class GmemAllocOpSizeOrder : public GmemAllocatorMethod {
public:
  struct OpAddr {
    ValueInfo op;
    int64_t start = 0;
    int64_t end = 0;
    uint32_t size = 0;
    uint32_t first_pos = 0;
    uint32_t end_pos = 0;

    OpAddr(ValueInfo _op, uint32_t _size, uint32_t _first_pos,
           uint32_t _end_pos) {
      op = _op;
      size = _size;
      first_pos = _first_pos;
      end_pos = _end_pos;
    }
  };
  // typedef std::list<std::shared_ptr<OpAddr>> LineSet;

public:
  GmemAllocOpSizeOrder(std::map<ValueInfo, int64_t> &gaddrMap,
                       uint32_t aligment);

  int64_t assignGaddr(std::vector<ValueInfo> &ops,
                      std::map<ValueInfo, TensorLive> &liveRange,
                      bool neuronMemoryReuse, int64_t baseGaddr) override;
};

class GmemAllocatorMethodFactory {
public:
  static GmemAllocatorMethod *makeMethod(std::string method_name,
                                         std::map<ValueInfo, int64_t> &gaddrMap,
                                         uint32_t aligment) {
    if (method_name == "FitFirstAssign") {
      return static_cast<GmemAllocatorMethod *>(
          new GmemAllocFitFirst(gaddrMap, aligment));
    } else if (method_name == "OpSizeOrderAssign") {
      return static_cast<GmemAllocatorMethod *>(
          new GmemAllocOpSizeOrder(gaddrMap, aligment));
    } else {
      assert(0);
      return nullptr;
    }
  }
};
} // namespace tpu
} // namespace tpu_mlir

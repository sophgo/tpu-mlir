//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/BMAddressAssign.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <set>
#include <sstream>
#include <tuple>
#include <vector>

using namespace llvm;
using namespace mlir;

using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {

void BMAddressAssign::assign(mlir::ModuleOp &module, bool reuse_addr) {
  int64_t alignment = BM168x::ALIGNMENT;
  int64_t start_addr = BM168x::CTX_START_ADDR;
  Builder builder(module.getContext());
  // assign weight first
  auto addr = start_addr;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](top::WeightOp op) {
      module::setAddress(op.getOutput(), addr);
      int64_t bytes = Arch::get_gmem_bytes(op.getOutput());
      addr = align_up(addr + bytes, alignment);
    });
  }
  module::setCoeffAddr(start_addr);
  module::setCoeffSize(addr - start_addr);
  // assign activation
  start_addr = addr;
  uint32_t loc = 0;
  // key: the operation pointer + output index, convert the result to type
  // int64_t
  std::map<ValueInfo, TensorLive> liveRange;
  std::map<Operation *, uint32_t> ops_loc;
  std::vector<ValueInfo> common_ops;
  std::vector<ValueInfo> inplace_ops;
  // 0.update liverange of ops and choose ops to allocate.
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      ops_loc[op] = loc;
      ++loc;
    });
  }
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (isa<ReturnOp>(op)) {
        updateLiveRangeofBMOps(op, 0, ops_loc, liveRange, common_ops,
                               inplace_ops, alignment);
      }
      int n = op->getNumResults();
      for (int i = 0; i < n; i++) {
        if (op->getResult(i).getType().isa<mlir::NoneType>()) {
          continue;
        }
        updateLiveRangeofBMOps(op, i, ops_loc, liveRange, common_ops,
                               inplace_ops, alignment);
      }
    });
  }
  // 1.assign common_ops
  // key: the operation pointer + output index, convert the result to type
  // int64_t
  std::map<ValueInfo, int64_t> gaddrMap;
  if (!common_ops.empty()) {
    GmemAllocator allocator(gaddrMap, alignment);
    auto gmemUsed =
        allocator.assignGaddr(common_ops, liveRange, reuse_addr, start_addr);
    addr += gmemUsed;
  }

  // 1.set common op address
  std::vector<ValueInfo> group_ops;
  for (auto &op_value : gaddrMap) {
    auto op = static_cast<Operation *>(op_value.first.op);
    module::setAddress(op->getResult(op_value.first.index), op_value.second);
    if (auto gOp = dyn_cast<tpu::GroupOp>(op)) {
      group_ops.emplace_back(op_value.first);
    }
  }

  // 2.set group op address
  for (auto &op_value : group_ops) {
    auto op = static_cast<Operation *>(op_value.op);
    if (auto gOp = dyn_cast<tpu::GroupOp>(op)) {
      auto &last_op = gOp.getBody().back().back();
      auto yield_op = dyn_cast<tpu::YieldOp>(last_op);
      assert(yield_op);
      int idx = 0;
      for (auto opd : yield_op.getOperands()) {
        auto addr = module::getAddress(gOp.getResult(idx));
        module::setAddress(opd, addr);
        idx++;
      }
    }
  }
  // 3.set inplace_ops address
  for (auto op : inplace_ops) {
    if (auto reshapeOp = dyn_cast<tpu::ReshapeOp>((Operation *)op.op)) {
      auto addr = module::getAddress(reshapeOp.getInput());
      module::setAddress(reshapeOp.getOutput(), addr);
    }
  }
  module::setNeuronAddr(start_addr);
  module::setNeuronSize(addr - start_addr);
  module::updateModuleTypes();
  module::setState(module::State::TPU_ADDRESSED);
}

void BMAddressAssign::updateLiveRangeofBMOps(
    Operation *op, int index, std::map<Operation *, uint32_t> &ops_loc,
    std::map<ValueInfo, TensorLive> &liveRange,
    std::vector<ValueInfo> &common_ops, std::vector<ValueInfo> &inplace_ops,
    int alignment) {
  auto updateOperandsLiveRange = [&](Operation *op, uint32_t endPosition) {
    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      auto operand = op->getOperand(i);
      auto opd = operand.getDefiningOp();
      if (opd == 0x0) {
        continue;
      }
      ValueInfo v_info(opd, operand.cast<OpResult>().getResultNumber());
      if (liveRange.find(v_info) != liveRange.end()) {
        if (isa<top::InputOp>(opd) && liveRange[v_info].end == 0xFFFFFFFF) {
          continue;
        }
        if (liveRange[v_info].end == 0xFFFFFFFF ||
            liveRange[v_info].end < endPosition) {
          liveRange[v_info].end = endPosition;
        }
      }
    }
  };
  ValueInfo v(op, index);
  uint32_t loc = ops_loc[op];
  uint32_t endPosition = loc + 1;
  if (isa<top::InputOp>(op)) {
    // liveRange.emplace_back(TensorLive(out, 0, 0xFFFFFFFF));
    uint32_t tensor_size = getTensorGmemSize(op, index, alignment);
    assert(liveRange.count(v) == 0);
    liveRange[v] = TensorLive(index, 0, 0xFFFFFFFF, tensor_size);
    updateOperandsLiveRange(op, endPosition);
    common_ops.emplace_back(v);
  } else if (isa<FuncOp, top::NoneOp, ReturnOp, top::WeightOp, func::CallOp,
                 tpu::YieldOp>(op) ||
             module::isOpInGroup(op)) {
    updateOperandsLiveRange(op, endPosition);
  } else if (isInPlaceOp(op)) {
    uint32_t maxPosition = endPosition;
    findInPlaceOpMaxUsePosition(op, maxPosition, ops_loc);
    updateOperandsLiveRange(op, maxPosition);
    inplace_ops.emplace_back(v);
  } else if (op->getDialect()->getNamespace() == "tpu") {
    uint32_t tensor_size = getTensorGmemSize(op, index, alignment);
    assert(liveRange.count(v) == 0);
    liveRange[v] = TensorLive(index, loc, 0xFFFFFFFF, tensor_size);
    updateOperandsLiveRange(op, endPosition);
    common_ops.emplace_back(v);
  } else {
    updateOperandsLiveRange(op, endPosition);
  }
}

void BMAddressAssign::findInPlaceOpMaxUsePosition(
    Operation *op, uint32_t &maxPosition,
    std::map<Operation *, uint32_t> &ops_loc) {
  for (auto &use : op->getResult(0).getUses()) {
    Operation *next = use.getOwner();
    if (isInPlaceOp(next)) {
      findInPlaceOpMaxUsePosition(next, maxPosition, ops_loc);
    } else {
      uint32_t curPosition = ops_loc[next] + 1;
      if (maxPosition < curPosition) {
        maxPosition = curPosition;
      }
    }
  }
}

bool BMAddressAssign::isInPlaceOp(Operation *op) {
  // TODO crop op
  if (isa<tpu::ReshapeOp>(op)) {
    return true;
  }
  return false;
}

int BMAddressAssign::getOutIndex(Operation *op, Value &out) {
  for (int i = 0; i < op->getNumResults(); i++) {
    if (op->getResult(i) == out) {
      return i;
    }
  }
  return -1;
}

uint32_t BMAddressAssign::getTensorGmemSize(Operation *op, int index,
                                            int64_t aligment_) {
  uint32_t size = Arch::get_gmem_bytes(op->getResult(index));
  // pad to aligment_
  if (size % aligment_) {
    size = size + aligment_ - (size % aligment_);
  }
  return size;
}

} // namespace tpu
} // namespace tpu_mlir

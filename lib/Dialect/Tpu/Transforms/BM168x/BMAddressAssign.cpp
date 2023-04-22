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

bool BMAddressAssign::is_next_subnet_input(Operation *op, int index) {
  bool ret = false;
  for (uint32_t i = 0; i < op->getNumOperands(); i++) {
    if (i == index) {
      for (const auto &user : op->getOperand(i).getUsers()) {
        if (isa<FuncOp>(user->getParentOp())) {
          FuncOp funcOp;
          funcOp = cast<FuncOp>(user->getParentOp());
          func::CallOp callee = module::getCallOp(funcOp);
          if (callee && std::distance(callee.getResult(index).user_begin(),
                                      callee.getResult(index).user_end())) {
            ret = true;
            break;
          }
        }
      }
    }
  }
  return ret;
}

void BMAddressAssign::assign(mlir::ModuleOp &module, bool reuse_addr) {
  int64_t alignment = BM168x::ALIGNMENT;
  int64_t start_addr = BM168x::CTX_START_ADDR;
  Builder builder(module.getContext());
  // assign weight first
  auto addr = start_addr;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](top::WeightOp op) {
      module::setAddress(op.getOutput(), addr);
      int64_t bytes = module::getBytes(op.getOutput());
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
  std::vector<Operation *> all_ops;
  // 0.update liverange of ops and choose ops to allocate.
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      ops_loc[op] = loc;
      ++loc;
      if (isa<FuncOp, top::NoneOp, top::WeightOp>(op) ||
          module::isOpInGroup(op)) {
        return;
      }
      all_ops.emplace_back(op);
    });
  }
  // update liverange from bottom to top.
  for (auto iter = all_ops.rbegin(); iter != all_ops.rend(); ++iter) {
    auto op = *iter;
    if (isa<ReturnOp>(op)) {
      updateLiveRangeofBMOps(op, 0, ops_loc, liveRange, common_ops, inplace_ops,
                             alignment);
    }
    int n = op->getNumResults();
    for (int i = 0; i < n; i++) {
      if (module::isNone(op->getResult(i))) {
        continue;
      }
      updateLiveRangeofBMOps(op, i, ops_loc, liveRange, common_ops, inplace_ops,
                             alignment);
    }
  }
  // 1.assign common_ops
  // key: the operation pointer + output index, convert the result to type
  // int64_t
  std::map<ValueInfo, int64_t> gaddrMap;
  if (!common_ops.empty()) {
    // FitFirstAssign should make sure op's start liverange ascendingly
    GmemAllocator::sortOpByLiveStart(common_ops, liveRange);
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

  // 2.set inplace_ops address
  // inplace_ops' order should be from input to output,thus reverse
  std::reverse(inplace_ops.begin(), inplace_ops.end());
  for (auto v_info : inplace_ops) {
    Operation *op = (Operation *)v_info.op;
    if (auto concatOp = dyn_cast<tpu::ConcatOp>(op)) {
      auto in0 = concatOp.getInputs()[0];
      if (auto rop = dyn_cast<tpu::ReshapeOp>(in0.getDefiningOp())) {
        in0 = rop.getInput();
      }
      int64_t addr = module::getAddress(in0);
      module::setAddress(concatOp.getOutput(), addr);
      int64_t offset = module::getBytes(in0);
      for (uint32_t i = 1; i < concatOp.getInputs().size(); i++) {
        auto input = concatOp.getInputs()[i];
        if (auto rop = dyn_cast<tpu::ReshapeOp>(input.getDefiningOp())) {
          module::setAddress(input, addr + offset);
          input = rop.getInput();
        }
        module::setAddress(input, addr + offset);
        offset += module::getBytes(input);
      }
    } else if (auto reshapeOp = dyn_cast<tpu::ReshapeOp>(op)) {
      auto addr = module::getAddress(reshapeOp.getInput());
      module::setAddress(reshapeOp.getOutput(), addr);
    } else if (auto sliceOp = dyn_cast<tpu::SliceOp>(op)) {
      auto addr = module::getAddress(sliceOp.getInput());
      auto p = sliceOp.parseParam();
      int axis;
      for (axis = 0; p.offset_4[axis] == 0 && axis < 4; axis++)
        ;
      size_t offset_bytes = 0;
      if (axis != 4) {
        offset_bytes =
            p.offset_4[axis] * module::getDtypeSize(sliceOp.getOutput());
        for (int i = axis + 1; i < 4; ++i) {
          offset_bytes *= p.is_4[i];
        }
      }
      module::setAddress(sliceOp.getOutput(), addr + offset_bytes);
    } else {
      llvm_unreachable("set address of undefined inplace op!");
    }
  }

  // 3.set group op address
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
      auto operand = module::getOperand(op, i);
      auto opd = operand.getDefiningOp();
      if (opd == 0x0 || isa<top::WeightOp, top::NoneOp>(opd)) {
        continue;
      }
      ValueInfo v_info(opd, operand.cast<OpResult>().getResultNumber());
      if (liveRange.find(v_info) != liveRange.end()) {
        // not first, update operand's liverange
        liveRange[v_info].start =
            std::min(liveRange[v_info].start, ops_loc[opd]);
        liveRange[v_info].end = std::max(liveRange[v_info].end, endPosition);
        liveRange[v_info].out_index = v_info.index;
      } else {
        // first update the operand, set its start, end, out_index and
        // tensor_size
        liveRange[v_info].start = ops_loc[opd];
        liveRange[v_info].end = endPosition;
        liveRange[v_info].out_index = v_info.index;
        liveRange[v_info].tensor_size =
            getTensorGmemSize(opd, v_info.index, alignment);
      }
      if (isa<top::InputOp>(opd)) {
        liveRange[v_info].start = 0;
        liveRange[v_info].end = 0xFFFFFFFF;
      }
    }
  };
  auto updateSOLOLiveRange = [&](Operation *op, ValueInfo v_info,
                                 uint32_t endPosition) {
    liveRange[v_info].start = ops_loc[op];
    liveRange[v_info].end = endPosition;
    liveRange[v_info].out_index = v_info.index;
    liveRange[v_info].tensor_size =
        getTensorGmemSize(op, v_info.index, alignment);
  };
  ValueInfo v(op, index);
  uint32_t loc = ops_loc[op];
  uint32_t endPosition = loc + 1;
  if (isa<top::InputOp>(op)) {
    // liveRange.emplace_back(TensorLive(out, 0, 0xFFFFFFFF));
    // updateOperandsLiveRange(op, endPosition);
    common_ops.emplace_back(v);
  } else if (isa<FuncOp, top::NoneOp, ReturnOp, top::WeightOp, func::CallOp,
                 tpu::YieldOp>(op) ||
             module::isOpInGroup(op)) {
    /* for multi_subnet, the returnOp's live range increase if it connect to
       next subnet Todo: other complex case need to handle, such as it connect
       to next func's inner group op */
    // simple solution: don;t set the endlife if it connect to next subnet
    // currently. updateOperandsLiveRange(op, endPosition+2); Here,
    // updateLiveRange from the last op to the first op, no need to concern it.
    updateOperandsLiveRange(op, endPosition);
  } else if (isInPlaceOp(op)) {
    if (isa<tpu::ConcatOp>(op)) {
      uint32_t tensor_size = getTensorGmemSize(op, index, alignment);
      // liveRange[v] = TensorLive(index, loc, 0xFFFFFFFF, 0);
      updateOperandsLiveRange(op, endPosition);
      std::vector<uint32_t> concatLive = getConcatOpLive(op, liveRange);
      for (int i = 0; i < op->getNumOperands(); ++i) {
        auto opd = module::getOperand(op, i);
        auto preOp = opd.getDefiningOp();
        if (auto rop = dyn_cast<tpu::ReshapeOp>(preOp)) {
          ValueInfo pre_v(preOp, opd.cast<OpResult>().getResultNumber());
          liveRange[pre_v].start = concatLive[0];
          liveRange[pre_v].end = concatLive[1];
          liveRange[pre_v].tensor_size = 0;
          opd = rop.getInput();
          preOp = opd.getDefiningOp();
        }
        ValueInfo pre_v(preOp, opd.cast<OpResult>().getResultNumber());
        liveRange[pre_v].start = concatLive[0];
        liveRange[pre_v].end = concatLive[1];
        if (i == 0) {
          liveRange[pre_v].tensor_size = tensor_size;
        } else {
          liveRange[pre_v].tensor_size = 0;
        }
      }
      inplace_ops.emplace_back(v);
    } else {
      uint32_t maxPosition = endPosition;
      findInPlaceOpMaxUsePosition(op, maxPosition, ops_loc);
      updateOperandsLiveRange(op, maxPosition);
      inplace_ops.emplace_back(v);
    }
  } else if (op->getDialect()->getNamespace() == "tpu") {
    ValueInfo cur_info(op, index);
    if (!module::isNone(op->getResult(index))) {
      if (liveRange.find(cur_info) == liveRange.end()) {
        updateSOLOLiveRange(op, cur_info, endPosition);
        common_ops.emplace_back(v);
        return;
      }
    }
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
  if (isa<tpu::ReshapeOp>(op)) {
    return true;
  } else if (auto sliceOp = dyn_cast<tpu::SliceOp>(op)) {
    auto p = sliceOp.parseParam();
    return p.fusible;
  } else if (auto concatOp = dyn_cast<tpu::ConcatOp>(op)) {
    return concatOp.getOnlyMerge();
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

std::vector<uint32_t>
BMAddressAssign::getConcatOpLive(Operation *op,
                                 std::map<ValueInfo, TensorLive> &liveRange) {
  // get concatOp and its operands' minimum start and maximum end as the whole
  // live.
  assert(isa<tpu::ConcatOp>(op));
  std::vector<uint32_t> live(2);
  ValueInfo op_info(op, 0);
  assert(liveRange.find(op_info) != liveRange.end());
  live[0] = liveRange[op_info].start;
  live[1] = liveRange[op_info].end;
  for (uint32_t i = 0; i < op->getNumOperands(); i++) {
    auto operand = op->getOperand(i);
    auto opd = operand.getDefiningOp();
    ValueInfo v_info(opd, operand.cast<OpResult>().getResultNumber());
    assert(liveRange.find(v_info) != liveRange.end());
    live[0] = std::min(liveRange[v_info].start, live[0]);
    live[1] = std::max(liveRange[v_info].end, live[1]);
  }
  return live;
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

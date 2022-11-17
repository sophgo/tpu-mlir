#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/CVAddressAssign.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <tuple>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {

void CVAddressAssign::assign(mlir::ModuleOp &module) {
  int64_t start_addr = (uint64_t)1 << 40;
  int64_t weight_alignment = 16;
  int64_t neuron_alignment = 64;
  Builder builder(module.getContext());
  // assign weight first
  auto addr = start_addr;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](top::WeightOp op) {
      Module::setAddress(op.output(), addr);
      int64_t bytes = Module::getBytes(op.output());
      addr = align_up(addr + bytes, weight_alignment);
    });
  }
  Module::setCoeffAddr(module, start_addr);
  Module::setCoeffSize(module, addr - start_addr);
  // key: the operation pointer & output index

  std::map<Operation *, uint32_t> ops_loc;
  std::map<ValueInfo, TensorLive> liveRange;
  std::vector<ValueInfo> inplace_ops;
  std::vector<ValueInfo> shared_outs;
  std::vector<std::vector<ValueInfo>> shared_outs_regions;
  std::vector<ValueInfo> private_outs;
  std::vector<ValueInfo> io_outs;

  // assign activation
  uint32_t loc = 0;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      ops_loc[op] = loc;
      ++loc;
    });
  }
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  Module::getInputsOutputs(module, inputs, outputs);

  // cal live range for each op
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (isa<FuncOp, top::NoneOp, func::ReturnOp, top::WeightOp, func::CallOp,
              tpu::YieldOp>(op)) {
        return;
      }
      int n = op->getNumResults();
      for (int i = 0; i < n; i++) {
        if (op->getResult(i).getType().isa<mlir::NoneType>()) {
          return;
        }
        updateLiveRangeOfOps(op, i, ops_loc, liveRange, inplace_ops,
                             neuron_alignment);
      }
    });
  }
  // determin the addr type
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      int n = op->getNumResults();
      for (int i = 0; i < n; i++) {
        if (op->getResult(i).getType().isa<mlir::NoneType>()) {
          continue;
        }
        ValueInfo v_info(op, i);
        if (liveRange.find(v_info) != liveRange.end()) {
          if (isOpBelongToIOMemoryRegion(op, i, outputs)) {
            if (io_outs.size() < 5) {
              io_outs.emplace_back(v_info);
            } else {
              private_outs.emplace_back(v_info);
            }
          } else if (isOpBelongToPrivateMemoryRegion(op, i)) {
            private_outs.emplace_back(v_info);
          } else {
            shared_outs.emplace_back(v_info);
          }
        }
      }
    });
    if (!shared_outs.empty()) {
      shared_outs_regions.emplace_back(std::move(shared_outs));
    }
  }

  int64_t sharedGmemOffset = 0;
  int64_t sharedGmemSize = 0;
  // key: the operation pointer & output index
  std::map<ValueInfo, int64_t> gaddrMap;

  for (auto &targetOuts : shared_outs_regions) {
    GmemAllocator allocator(gaddrMap, neuron_alignment);
    auto gmemUsed =
        allocator.assignGaddr(targetOuts, liveRange, true, sharedGmemOffset);
    if (sharedGmemSize < sharedGmemOffset + gmemUsed) {
      sharedGmemSize = sharedGmemOffset + gmemUsed;
    }
  }

  int64_t baseGaddr = (((uint64_t)2) << 40);
  int64_t privateGmemSize = 0;
  // 2. Assign gaddr for ops in private region.
  if (!private_outs.empty()) {
    GmemAllocator allocator(gaddrMap, neuron_alignment);
    privateGmemSize =
        allocator.assignGaddr(private_outs, liveRange, true, baseGaddr);
  }

  // 3. Assign gaddr for ops in IO memory regin.
  for (int i = 0; i < (int)io_outs.size(); ++i) {
    gaddrMap[io_outs[i]] = (((uint64_t)3 + i) << 40);
  }
  // 4. set addr according to gaddrMap
  for (auto &op_addr : gaddrMap) {
    Operation *op = static_cast<Operation *>(op_addr.first.op);
    Module::setAddress(op->getResult(op_addr.first.index), op_addr.second);
  }
  for (auto &v_info : inplace_ops) {
    updateAddressOfInPlaceOp(v_info);
  }

  // TODO markGmemReusedOp
  // TODO crop concat pattern

  Module::setNeuronSize(module, sharedGmemSize);
  Module::setGmemPrivateSize(module, privateGmemSize);
  Module::updateModuleTypes(module);
  Module::setState(module, Module::State::TPU_ADDRESSED);
}

bool CVAddressAssign::isOpBelongToIOMemoryRegion(Operation *op, int index,
                                                 std::vector<Value> &outputs) {
  // Warning, IO memory region can only has capacity to store 5 ops.
  if (isa<top::InputOp>(op)) {
    return true;
  }
  if (isOutput(op, index)) {
    auto value = op->getResult(index);
    if (std::find(outputs.begin(), outputs.end(), value) != outputs.end()) {
      return true;
    }
  }
  return false;
}

bool CVAddressAssign::isOpBelongToPrivateMemoryRegion(Operation *op,
                                                      int index) {
  if (isa<top::InputOp>(op) || isOutput(op, index) ||
      isa<tpu::GenericCpuOp>(op) ||
      isInPlaceOpBelongToPrivateMemoryRegion(op, index)) {
    return true;
  }
  return false;
}

bool CVAddressAssign::isInPlaceOpBelongToPrivateMemoryRegion(Operation *op,
                                                             int index) {
  for (auto &use : op->getResult(index).getUses()) {
    Operation *next = use.getOwner();
    if (isInPlaceOp(next) && isOutput(next, 0)) {
      return true;
    }
  }
  return false;
}

void CVAddressAssign::updateLiveRangeOfOps(
    Operation *op, int index, std::map<Operation *, uint32_t> &ops_loc,
    std::map<ValueInfo, TensorLive> &liveRange,
    std::vector<ValueInfo> &inplace_ops, int64_t alignment) {
  auto updateOperandsLiveRange = [&](Operation *op, uint32_t endPosition) {
    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      auto operand = Module::getOperand(op, i);
      auto opd = operand.getDefiningOp();
      if (opd == 0x0) {
        assert(0);
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

  ValueInfo value_info(op, index);
  uint32_t loc = ops_loc[op];
  uint32_t endPosition = loc + 1;
  // TODO refer to tpu_compiler AssignNeuronAddress.cpp
  if (isa<top::InputOp>(op)) {
    uint32_t tensor_size = getTensorGmemSize(op, index, alignment);
    assert(liveRange.count(value_info) == 0);
    // TensorLive tl = TensorLive(index, 0, 0xFFFFFFFF, tensor_size);
    liveRange[value_info] = TensorLive(index, 0, 0xFFFFFFFF, tensor_size);
  } else if (Module::isOpInGroup(op)) {
    updateOperandsLiveRange(op, endPosition);
  } else if (isInPlaceOp(op)) {
    uint32_t maxPosition = endPosition;
    uint32_t tensor_size = getTensorGmemSize(op, index, alignment);
    findInPlaceOpMaxUsePosition(op, maxPosition, ops_loc);
    updateOperandsLiveRange(op, maxPosition);
    updateLiveRangeOfInPlaceOp(op, index, liveRange, loc, maxPosition,
                               tensor_size);
    inplace_ops.emplace_back(op, index);
  } else if (op->getDialect()->getNamespace() == "tpu") {
    uint32_t tensor_size = getTensorGmemSize(op, index, alignment);
    assert(liveRange.count(value_info) == 0);
    // TensorLive tl = TensorLive(index, loc, 0xFFFFFFFF, tensor_size);
    liveRange[value_info] = TensorLive(index, loc, 0xFFFFFFFF, tensor_size);
    updateOperandsLiveRange(op, endPosition);
  } else {
    updateOperandsLiveRange(op, endPosition);
  }
}

void CVAddressAssign::updateLiveRangeOfInPlaceOp(
    Operation *op, int i, std::map<ValueInfo, TensorLive> &liveRange,
    int64_t start, int64_t end, uint32_t tensor_size) {
  if (auto concatOp = dyn_cast<tpu::ConcatOp>(op)) {
    // For ConcatN. To solve concat opt when axis = 0,
    // it need the operand should be continuous global memory.
    int64_t min_start = start;
    int64_t max_end = end;
    auto min_start_op = ValueInfo(op, i);
    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      auto operand = Module::getOperand(op, i);
      auto opd = operand.getDefiningOp();
      assert(opd != 0x0);
      ValueInfo opd_info(opd, operand.cast<OpResult>().getResultNumber());
      assert(liveRange.find(opd_info) != liveRange.end());
      assert(max_end >= liveRange[opd_info].end);
      if (liveRange[opd_info].start < min_start) {
        if (liveRange.find(min_start_op) != liveRange.end()) {
          liveRange.erase(min_start_op);
        }
        min_start_op = opd_info;
        min_start = liveRange[min_start_op].start;
        liveRange[min_start_op].end = max_end;
        liveRange[min_start_op].tensor_size = tensor_size;
      } else {
        liveRange.erase(opd_info);
      }
    }
  }
}

void CVAddressAssign::updateAddressOfInPlaceOp(ValueInfo &v_info) {
  auto op = static_cast<Operation *>(v_info.op);
  if (auto concatOp = dyn_cast<tpu::ConcatOp>(op)) {
    int64_t base_addr = -1;
    uint32_t nof_found = 0;
    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      auto operand = Module::getOperand(op, i);
      if (operand.getType().cast<RankedTensorType>().getEncoding()) {
        base_addr = Module::getAddress(operand);
        ++nof_found;
      }
    }
    assert(nof_found == 1);
    int64_t offset = 0;
    Module::setAddress(op->getResult(0), base_addr + offset);
    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      auto operand = Module::getOperand(op, i);
      auto opd = operand.getDefiningOp();
      if (opd == 0x0) {
        continue;
      }
      int this_index = operand.cast<OpResult>().getResultNumber();
      // assert(liveRange[save_opd].tensor_size == 0);
      uint32_t tensor_size = Module::getBytes(opd->getResult(this_index));
      Module::setAddress(opd->getResult(this_index), base_addr + offset);
      offset += tensor_size;
    }
  } else if (auto reshapeOp = dyn_cast<tpu::ReshapeOp>(op)) {
    Module::setAddress(reshapeOp.output(),
                       Module::getAddress(reshapeOp.input()));
  } else if (auto sliceOp = dyn_cast<tpu::SliceOp>(op)) {
    std::vector<int64_t> i_s;
    std::vector<int64_t> o_s;
    std::vector<int> offset_4;
    std::vector<int> step_4;
    bool fusible = false;
    sliceOp.parseParam(i_s, o_s, offset_4, step_4, fusible);
    int axis;
    for (axis = 0; offset_4[axis] == 0 && axis < 4; axis++)
      ;
    size_t offset_bytes = 0;
    if (axis != 4) {
      offset_bytes = offset_4[axis] * Module::getDtypeSize(sliceOp.output());
      for (int i = axis + 1; i < 4; ++i) {
        offset_bytes *= i_s[i];
      }
    }
    Module::setAddress(sliceOp.output(),
                       Module::getAddress(sliceOp.input()) + offset_bytes);
  } else {
    llvm_unreachable("set address of undefined inplace op!");
  }
}

bool CVAddressAssign::isInPlaceOp(Operation *op) {
  if (isa<tpu::ReshapeOp>(op)) {
    return true;
  } else if (isa<tpu::ConcatOp>(op)) {
    auto concat_op = dyn_cast<tpu::ConcatOp>(op);
    if (concat_op.only_merge()) {
      return true;
    }
  } else if (isa<tpu::SliceOp>(op)) {
    auto slice_op = dyn_cast<tpu::SliceOp>(op);
    std::vector<int64_t> i_s;
    std::vector<int64_t> o_s;
    std::vector<int> offset_4;
    std::vector<int> step_4;
    bool fusible = false;
    slice_op.parseParam(i_s, o_s, offset_4, step_4, fusible);
    if (fusible) {
      return true;
    }
  } else {
    return false;
  }
  return false;
}

bool CVAddressAssign::isOutput(Operation *op, int index) {
  for (auto &use : op->getResult(index).getUses()) {
    Operation *next = use.getOwner();
    if (isa<func::ReturnOp>(next)) {
      return true;
    }
  }
  return false;
}

void CVAddressAssign::findInPlaceOpMaxUsePosition(
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

uint32_t CVAddressAssign::getTensorGmemSize(Operation *op, int index,
                                            int64_t aligment_) {
  uint32_t size = Module::getBytes(op->getResult(index));
  // pad to aligment_
  if (size % aligment_) {
    size = size + aligment_ - (size % aligment_);
  }
  return size;
}
} // namespace tpu
} // namespace tpu_mlir

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/GmemAllocator.hpp"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Format.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include <tuple>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {
// for cv18xx. Todo let cv18xx bm168x use same addrAssigner
class CVAddressAssignPass : public CVAddressAssignBase<CVAddressAssignPass> {
public:

  CVAddressAssignPass() {}
  void runOnOperation() override {
    auto module = getOperation();
    auto state = Module::getState(module);
    if (state != Module::State::TPU_DIVIDED) {
      llvm_unreachable("module should be divided");
    }
    Module::removeUnusedOp(module);
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
    // assign activation
    uint32_t loc = 0;
    //key: the operation pointer + output index, convert the result to type int64_t
    std::map<int64_t, TensorLive> liveRange;
    std::map<Operation *, uint32_t> ops_loc;
    std::vector<int64_t> shared_outs;
    std::vector<std::vector<int64_t>> shared_outs_regions;
    std::vector<int64_t> private_outs;
    std::vector<int64_t> io_outs;
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        ops_loc[op] = loc;
        ++loc;
      });
    }

    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        if (isa<FuncOp, top::NoneOp, func::ReturnOp, top::WeightOp,
                func::CallOp, tpu::YieldOp>(op)) {
          return;
        }
        int n = op->getNumResults();
        for (int i = 0; i < n; i++) {
          bool chosen = false;
          updateLiveRangeOfOps(op, i, ops_loc, liveRange, chosen, neuron_alignment);
          if (chosen) {
            if (isOpBelongToIOMemoryRegion(op)) {
              if (io_outs.size() < 5) {
                io_outs.emplace_back((int64_t)op + i);
              } else {
                private_outs.emplace_back((int64_t)op + i);
              }
            } else if (isOpBelongToSharedMemoryRegion(op)) {
              shared_outs.emplace_back((int64_t)op + i);
            } else {
              private_outs.emplace_back((int64_t)op + i);
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
    //key: the operation pointer + output index, convert the result to type int64_t
    std::map<int64_t, int64_t> gaddrMap;

    for (auto &targetOuts : shared_outs_regions) {
      GmemAllocator allocator(gaddrMap, neuron_alignment);
      auto gmemUsed =
          allocator.assignGaddr(targetOuts, liveRange, true, sharedGmemOffset);
      if (sharedGmemSize < sharedGmemOffset + gmemUsed) {
        sharedGmemSize = sharedGmemOffset + gmemUsed;
      }
    }
    // To solve concat opt when axis = 0, it need the operand should be
    // continuous global memory.
    /*
    auto assignedGaddr = sharedGmemSize;
    for (auto &targetOps : opsInSharedMemoryRegions) {
      for (auto &op : targetOps) {
        auto size  = GmemAllocator::assignSpecifiedGmemToOp(op, gaddrMap,
                                                       assignedGaddr,
                                                       clNeuronAlignment);
        assignedGaddr += size;
      }
    }
    sharedGmemSize = assignedGaddr ;
    */

    int64_t baseGaddr = (((uint64_t)2) << 40);
    int64_t privateGmemSize = 0;
    // 2. Assign gaddr for ops in private region.
    if (!private_outs.empty()) {
      GmemAllocator allocator(gaddrMap, neuron_alignment);
      privateGmemSize = allocator.assignGaddr(private_outs, liveRange,
                                              true, baseGaddr);
    }

    // 3. Assign gaddr for ops in IO memory regin.
    for (int i = 0; i < (int)io_outs.size(); ++i) {
      gaddrMap[io_outs[i]] = (((uint64_t)3 + i) << 40);
    }

    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        int n = op->getNumResults();
        for (int i = 0; i < n; i++) {
          int64_t save_opd = (int64_t)op + i;
          if (gaddrMap.find(save_opd) != gaddrMap.end()) {
            if (!isa<tpu::ReshapeOp>(op)) {
              Module::setAddress(op->getResult(i), gaddrMap[save_opd]);
            }
          } else {
            if (isa<tpu::ReshapeOp>(op)) {
              auto reshapeOp = dyn_cast<tpu::ReshapeOp>(op);
              Module::setAddress(reshapeOp.output(), Module::getAddress(reshapeOp.input()));
            }
          }
        }
      });
    }

    // TODO markGmemReusedOp
    // TODO crop concat pattern

    Module::setNeuronSize(module, sharedGmemSize);
    Module::setGmemPrivateSize(module, privateGmemSize);
    Module::updateModuleTypes(module);
    Module::setState(module, Module::State::TPU_ADDRESSED);
  }

protected:
  bool isOpBelongToIOMemoryRegion(Operation *op) {
    // Warning, IO memory region can only has capacity to store 5 ops.
    if (isa<top::InputOp>(op)) {
      return true;
    } else {
      auto next = getNextOp(op);
      if (next && isa<ReturnOp>(next)) {
        return true;
      }
    }
    return false;
  }

  bool isOpBelongToSharedMemoryRegion(Operation *op) {
    if (isOutputOp(op)) {
      return false;
    }
    for (auto &use : op->getResult(0).getUses()) {
      Operation *next = use.getOwner();
      // TODO need traverse next until not InPlaceOp
      if (isInPlaceOp(next) && isOutputOp(next)) {
        return false;
      }
    }
    return true;
  }

  void updateLiveRangeOfOps(
      Operation *op, int index, std::map<Operation *, uint32_t> &ops_loc,
      std::map<int64_t, TensorLive> &liveRange,
      bool &chosen, int64_t alignment = 64) {
    auto updateOperandsLiveRange = [&](Operation *op, uint32_t endPosition) {
      for (uint32_t i = 0; i < op->getNumOperands(); i++) {
        auto operand = op->getOperand(i);
        auto opd = operand.getDefiningOp();
        if (opd == 0x0) {
          continue;
        }
        int64_t save_opd = (int64_t)opd;
        if (opd->getNumResults() > 1) {
          int this_index = getOutIndex(opd, operand);
          assert(this_index != -1);
          save_opd = save_opd + this_index;
        }
        if (liveRange.find(save_opd) != liveRange.end()) {
          if (isa<top::InputOp>(opd) && liveRange[save_opd].end == 0xFFFFFFFF) {
            continue;
          }
          if (liveRange[save_opd].end == 0xFFFFFFFF ||
              liveRange[save_opd].end < endPosition) {
            liveRange[save_opd].end = endPosition;
          }
        }
      }
    };

    auto save_op = (int64_t)op + index;
    uint32_t loc = ops_loc[op];
    uint32_t endPosition = loc + 1;
    // TODO refer to tpu_compiler AssignNeuronAddress.cpp
    if (isa<top::InputOp>(op)) {
      uint32_t tensor_size = getTensorGmemSize(op, index, alignment);
      assert(liveRange.count(save_op) == 0);
      //TensorLive tl = TensorLive(index, 0, 0xFFFFFFFF, tensor_size);
      liveRange[save_op] = TensorLive(index, 0, 0xFFFFFFFF, tensor_size);
      chosen = true;
    } else if (Module::isOpInGroup(op)) {
      updateOperandsLiveRange(op, endPosition);
      chosen = false;
    } else if (isInPlaceOp(op)) {
      uint32_t maxPosition = endPosition;
      findInPlaceOpMaxUsePosition(op, maxPosition, ops_loc);
      updateOperandsLiveRange(op, maxPosition);
      chosen = false;
    } else if (op->getDialect()->getNamespace() == "tpu") {
      uint32_t tensor_size = getTensorGmemSize(op, index, alignment);
      assert(liveRange.count(save_op) == 0);
      //TensorLive tl = TensorLive(index, loc, 0xFFFFFFFF, tensor_size);
      liveRange[save_op] = TensorLive(index, loc, 0xFFFFFFFF, tensor_size);
      updateOperandsLiveRange(op, endPosition);
      chosen = true;
    } else {
      updateOperandsLiveRange(op, endPosition);
      chosen = false;
    }
  }

  bool isInPlaceOp(Operation *op) {
    // TODO crop op
    if (isa<tpu::ReshapeOp>(op)) {
      return true;
    }
    return false;
  }

  bool isOutputOp(Operation *op) {
    for (auto &use : op->getResult(0).getUses()) {
      Operation *next = use.getOwner();
      if (isa<func::ReturnOp>(next)) {
        return true;
      }
    }
    return false;
  }

  Operation *getNextOp(Operation *op) {
    Operation *nextOp = nullptr;
    if (op->getResult(0).hasOneUse()) {
      for (auto &use : op->getResult(0).getUses()) {
        nextOp = use.getOwner();
        break;
      }
      assert(nextOp && "nextOp is nullptr");
    }
    // if not found, will return NULL
    return nextOp;
  }

  void findInPlaceOpMaxUsePosition(Operation *op, uint32_t &maxPosition, std::map<Operation *, uint32_t> &ops_loc) {
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

  int getOutIndex(Operation *op, Value &out) {
    for (int i = 0; i < op->getNumResults(); i++) {
      if (op->getResult(i) == out) {
        return i;
      }
    }
    return -1;
  }

  uint32_t getTensorGmemSize(Operation *op, int index, int64_t aligment_) {
    uint32_t size = Module::getBytes(op->getResult(index));
    // pad to aligment_
    if (size % aligment_) {
      size = size + aligment_ - (size % aligment_);
    }
    return size;
  }

};

std::unique_ptr<OperationPass<ModuleOp>> createCVAddressAssignPass() {
  return std::make_unique<CVAddressAssignPass>();
}
} // namespace tpu
} // namespace tpu_mlir

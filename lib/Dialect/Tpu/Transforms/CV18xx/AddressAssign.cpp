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
  //Value, offset, size, ref_cnt
  using gmem_entry = std::tuple<mlir::Value, int64_t, int64_t, int64_t>;

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
    std::map<Operation *, std::vector<uint32_t>> liveRange;
    std::map<Operation *, uint32_t> ops_loc;
    std::vector<Operation *> shared_ops;
    std::vector<std::vector<Operation *> > shared_ops_regions;
    std::vector<Operation *> private_ops;
    std::vector<Operation *> io_ops;

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
        bool chosen = false;
        updateLiveRangeOfOps(op, ops_loc, liveRange, chosen);
        printf("op name:%s  func:%s loc:%u\n",
               Module::getName(op).str().c_str(), func.getName().str().c_str(),
               ops_loc[op]);
        if (chosen) {
          if (isOpBelongToIOMemoryRegion(op)) {
            if (io_ops.size() < 5) {
              io_ops.emplace_back(op);
              printf("op name:%s insert iomem\n",
                     Module::getName(op).str().c_str());
            } else {
              private_ops.emplace_back(op);
              printf("op name:%s insert primem\n",
                     Module::getName(op).str().c_str());
            }
          } else if (isOpBelongToSharedMemoryRegion(op)) {
            shared_ops.emplace_back(op);
            printf("op name:%s insert shared mem\n",
                   Module::getName(op).str().c_str());
          } else {
            private_ops.emplace_back(op);
            printf("op name:%s insert primem\n",
                   Module::getName(op).str().c_str());
          }
        }
      });
      if (!shared_ops.empty()) {
        shared_ops_regions.emplace_back(std::move(shared_ops));
      }
    }

    // assign gaddr for ops in different memory regions
    int64_t sharedGmemOffset = 0;
    int64_t sharedGmemSize = 0;
    std::map<Operation *, int64_t> gaddrMap;

    for (auto &targetOps : shared_ops_regions) {
      GmemAllocator allocator(gaddrMap, neuron_alignment);
      auto gmemUsed =
          allocator.assignGaddr(targetOps, liveRange, true, sharedGmemOffset);
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
    if (!private_ops.empty()) {
      GmemAllocator allocator(gaddrMap, neuron_alignment);
      privateGmemSize = allocator.assignGaddr(private_ops, liveRange,
                                              true, baseGaddr);
    }

    // 3. Assign gaddr for ops in IO memory regin.
    for (int i = 0; i < (int)io_ops.size(); ++i) {
      gaddrMap[io_ops[i]] = (((uint64_t)3 + i) << 40);
    }

    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        if (gaddrMap.find(op) != gaddrMap.end()) {
          if (!isa<tpu::ReshapeOp>(op)) {
            Module::setAddress(op->getResult(0), gaddrMap[op]);
          }
        }
      });
    }

    // TODO markGmemReusedOp
    // TODO crop concat pattern

    Module::setNeuronAddr(module, sharedGmemOffset);
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
      Operation *op, std::map<Operation *, uint32_t> &ops_loc,
      std::map<Operation *, std::vector<uint32_t>> &liveRange, bool &chosen) {
    auto updateOperandsLiveRange = [&](Operation *op, uint32_t endPosition) {
      for (uint32_t i = 0; i < op->getNumOperands(); i++) {
        auto opd = op->getOperand(i).getDefiningOp();
        if (liveRange.find(opd) != liveRange.end()) {
          if (isa<top::InputOp>(opd) && liveRange[opd][1] == 0xFFFFFFFF) {
            continue;
          }
          if (liveRange[opd][1] == 0xFFFFFFFF ||
              liveRange[opd][1] < endPosition) {
            liveRange[opd][1] = endPosition;
          }
        }
      }
    };
    uint32_t endPosition = ops_loc[op] + 1;
    uint32_t loc = ops_loc[op];
    // TODO refer to tpu_compiler AssignNeuronAddress.cpp
    if (isa<top::InputOp>(op)) {
      liveRange[op] = {0, 0xFFFFFFFF};
      updateOperandsLiveRange(op, endPosition);
      chosen = true;
      printf("op name:%s case input\n", Module::getName(op).str().c_str());
    } else if (isInPlaceOp(op)) {
      uint32_t maxPosition = endPosition;
      findInPlaceOpMaxUsePosition(op, maxPosition, ops_loc);
      updateOperandsLiveRange(op, maxPosition);
      printf("op name:%s case inplace\n", Module::getName(op).str().c_str());
    } else if (fuse_address(op) && !isa<tpu::StoreOp>(op)) {
      updateOperandsLiveRange(op, endPosition);
      chosen = false;
      printf("op name:%s case fuse\n", Module::getName(op).str().c_str());
    } else if (op->getDialect()->getNamespace() == "tpu") {
      liveRange[op] = {loc, 0xFFFFFFFF};
      updateOperandsLiveRange(op, endPosition);
      chosen = true;
      printf("op name:%s case tpu\n", Module::getName(op).str().c_str());
    } else {
      updateOperandsLiveRange(op, endPosition);
      chosen = false;
      printf("op name:%s case other\n", Module::getName(op).str().c_str());
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
      if (isa<func::ReturnOp>(op)) {
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

  bool fuse_address(Operation *op) {
    if (Module::isOpInGroup(op)) {
      return true;
    }
    if (auto reshapeOp = dyn_cast<tpu::ReshapeOp>(op)) {
      if (chip == Module::Chip::BM1684x) {
        auto addr = Module::getAddress(reshapeOp.input());
        Module::setAddress(reshapeOp.output(), addr);
        return true;
      }
    }
    return false;
  }

  void findInPlaceOpMaxUsePosition(Operation *op, uint32_t &maxPosition, std::map<Operation *, uint32_t> &ops_loc) {
    for (auto &use : op->getResult(0).getUses()) {
      Operation *next = use.getOwner();
      if (isInPlaceOp(next)) {
        findInPlaceOpMaxUsePosition(next, maxPosition, ops_loc);
      } else {
        uint32_t curPosition = ops_loc[op] + 1;
        if (maxPosition < curPosition) {
          maxPosition = curPosition;
        }
      }
    }
  }

  StringRef chip;
private:
   //record the allocated Gmem:Value, offset, size, ref_cnt
  std::vector<gmem_entry> rec_tbl;
  std::vector<mlir::Value> hold_edges;
  std::set<int64_t> in_using_addr;
};

std::unique_ptr<OperationPass<ModuleOp>> createCVAddressAssignPass() {
  return std::make_unique<CVAddressAssignPass>();
}
} // namespace tpu
} // namespace tpu_mlir

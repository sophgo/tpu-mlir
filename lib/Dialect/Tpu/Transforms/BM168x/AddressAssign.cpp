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
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Backend/BM168x/BM1684x.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
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

class AddressAssignPass : public AddressAssignBase<AddressAssignPass> {
public:
  //Value, offset, size, ref_cnt
  using gmem_entry = std::tuple<mlir::Value, int64_t, int64_t, int64_t>;

  AddressAssignPass() {}
  void runOnOperation() override {
    auto module = getOperation();
    auto state = Module::getState(module);
    if (state != Module::State::TPU_DIVIDED) {
      llvm_unreachable("module should be divided");
    }
    Module::removeUnusedOp(module);

    int64_t start_addr = 0;
    int64_t alignment = BM168x::ALIGNMENT;
    chip = Module::getChip(module);
    if (chip == Module::Chip::BM1684) {
      start_addr = BM1684::instance().get_ctx_start_addr();
    } else if (chip == Module::Chip::BM1684x) {
      start_addr = BM1684x::instance().get_ctx_start_addr();
    } else {
      llvm_unreachable("chip not support now");
    }
    Builder builder(module.getContext());
    // assign weight first
    auto addr = start_addr;
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](top::WeightOp op) {
        Module::setAddress(op.output(), addr);
        int64_t bytes = Module::getBytes(op.output());
        addr = align_up(addr + bytes, alignment);
      });
    }
    Module::setCoeffAddr(module, start_addr);
    Module::setCoeffSize(module, addr - start_addr);
    // assign activation
    start_addr = addr;
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        //adjust the lifecycle.
        calc_tensor_life_cycle(op);
        if (isa<FuncOp, top::NoneOp, func::ReturnOp, top::WeightOp,
                func::CallOp, tpu::YieldOp>(op)) {
        } else if (fuse_address(op)) {
          if (isa<tpu::ReshapeOp>(op)) {
            //need to record the lifecycle and allocated addr of output tensor
            for (auto out : op->getResults()) {
              int user_size = 0;
              for (auto user : out.getUsers())
                user_size++;
              record_info(out, Module::getAddress(out), user_size);
            }
          }
        } else {
          for (auto out : op->getResults()) {
            int64_t bytes = Module::getBytes(out);

            //check if can reuse the gmem according to the lifecycle of tensor.
            int64_t reused_addr = 0;
            check_can_reuse_gmem(op, bytes, reused_addr);

            //use the number of tensor's Users as the life cycle.
            int user_size = 0;
            for (auto user : out.getUsers()) {
              user_size++;
            }

            /* don't share the gmem between input/output tensor except reshape op .*/
            if (reused_addr == 0) {
              Module::setAddress(out, addr);
              record_info(out, addr, user_size);
              addr = align_up(addr + bytes, alignment);
            } else {
              Module::setAddress(out, reused_addr);
              record_info(out, reused_addr, user_size);
            }
            adjust_eol_edge(op);
          }
        }
        //adjust_eol_edge(op);
      });
      // sync StoreOp addr
      func.walk([&](tpu::GroupOp gOp) {
        int idx = 0;
        gOp.body().walk([&](tpu::StoreOp sOp) {
          auto addr = Module::getAddress(gOp.getResult(idx));
          Module::setAddress(sOp.output(), addr);
          idx++;
        });
      });
    }
    Module::setNeuronAddr(module, start_addr);
    Module::setNeuronSize(module, addr - start_addr);
    Module::updateModuleTypes(module);
    Module::setState(module, Module::State::TPU_ADDRESSED);
  }

protected:
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

  void calc_tensor_life_cycle(Operation *op) {
    //decrease the ref_cnt of allocated tensor for further reuse gmem
    for (auto v: op->getOperands()) {
      //change the ref_cnt
      for (auto it = rec_tbl.begin(); it != rec_tbl.end(); it++) {
        if (std::get<0>(*it) == v && std::get<3>(*it) >= 1 ) {
          //decrease the ref_cnt
          std::get<3>(*it)--;
        }
      }
    }
  }

  void adjust_eol_edge(Operation *op) {
    //remove the holdtensor
    for (auto v: op->getOperands()) {
      auto iter = std::find(std::begin(hold_edges), std::end(hold_edges), v);
      if (iter != std::end(hold_edges)) {
        for (auto it = rec_tbl.begin(); it != rec_tbl.end(); it++) {
          if (std::get<0>(*it) == v && std::get<3>(*it) == 0 ) {
            //remove the using addr
            in_using_addr.erase(std::get<1>(*it));
            hold_edges.erase(std::remove(hold_edges.begin(), hold_edges.end(), v), hold_edges.end());
          }
        }
      }
    }
  }

  int is_same_addr_with_input(Operation *op,
                              int64_t addr) {
    int64_t same = 0;
    for (auto in: op->getOperands())
    {
      if (in.getType().isa<RankedTensorType>()
          && in.getType().cast<RankedTensorType>().getEncoding()
          && in.getType().cast<RankedTensorType>().getEncoding().isa<IntegerAttr>()
          && Module::getAddress(in) == addr) {
        same = 1;
        break;
      }
    }
    return same;
  }

  void check_can_reuse_gmem(Operation *op,
                            int64_t need_size,
                            int64_t &reused_addr) {
    for (auto iter = rec_tbl.begin(); iter != rec_tbl.end(); iter++) {
      //TODO: merge more than two EOL tensor's consecutive gmem to one gmem
      auto it = std::find(std::begin(hold_edges), std::end(hold_edges), std::get<0>(*iter));
      if (it == hold_edges.end()
          && !in_using_addr.count(std::get<1>(*iter))
          && std::get<3>(*iter) == 0
            && std::get<2>(*iter) >= need_size) {
        if (!is_same_addr_with_input(op, std::get<1>(*iter))) {
          reused_addr = std::get<1>(*iter);
          break;
        }
      }
    }
  }

  void record_info(mlir::Value tensor,
                   int64_t offset,
                   int64_t users){
    rec_tbl.push_back(std::make_tuple(tensor, offset, Module::getBytes(tensor), users));
    hold_edges.push_back(tensor);
    in_using_addr.insert(offset);
  }
  StringRef chip;
private:
   //record the allocated Gmem:Value, offset, size, ref_cnt
  std::vector<gmem_entry> rec_tbl;
  std::vector<mlir::Value> hold_edges;
  std::set<int64_t> in_using_addr;
};

std::unique_ptr<OperationPass<ModuleOp>> createAddressAssignPass() {
  return std::make_unique<AddressAssignPass>();
}
} // namespace tpu
} // namespace tpu_mlir

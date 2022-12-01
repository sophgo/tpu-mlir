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
  std::map<ValueInfo, OpElement> op_infos;
  std::vector<Operation *> ops;
  std::vector<ValueInfo> inplace_ops;
  std::map<std::string, std::vector<ValueInfo>> shared_outs_regions;
  std::vector<ValueInfo> private_outs;
  std::vector<ValueInfo> io_outs;

  // assign activation
  uint32_t loc = 0;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      ops_loc[op] = loc;
      ++loc;
      if (isa<FuncOp, top::NoneOp, top::WeightOp, func::CallOp, tpu::YieldOp>(
              op)) {
        return;
      }
      ops.emplace_back(op);
    });
  }
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  Module::getInputsOutputs(module, inputs, outputs);
  for (auto iter = ops.rbegin(); iter != ops.rend(); ++iter) {
    updateLiveRange(*iter, ops_loc, op_infos, outputs, neuron_alignment);
  }

  for (auto iter = ops.begin(); iter != ops.end(); ++iter) {
    auto op = *iter;
    int n = op->getNumResults();
    for (int i = 0; i < n; ++i) {
      if (op->getResult(i).getType().isa<mlir::NoneType>()) {
        continue;
      }
      ValueInfo v_info(op, i);
      assert(op_infos.find(v_info) != op_infos.end());
      if (op_infos[v_info].need_alloc) {
        liveRange[v_info] = op_infos[v_info].live;
        switch (op_infos[v_info].mem_type) {
        case MEM_IOMEM:
          if (io_outs.size() < 5) {
            io_outs.emplace_back(v_info);
          } else {
            private_outs.emplace_back(v_info);
          }
          break;
        case MEM_PRIVATE:
          private_outs.emplace_back(v_info);
          break;
        case MEM_SHARED:
          auto func_name = dyn_cast<FuncOp>(op->getParentOp()).getName().str();
          shared_outs_regions[func_name].emplace_back(v_info);
          break;
        }
      } else {
        if (op_infos[v_info].inplace) {
          inplace_ops.emplace_back(v_info);
        }
      }
    }
  }

  int64_t sharedGmemOffset = 0;
  int64_t sharedGmemSize = 0;
  // key: the operation pointer & output index
  std::map<ValueInfo, int64_t> gaddrMap;

  for (auto &targetOuts : shared_outs_regions) {
    GmemAllocator allocator(gaddrMap, neuron_alignment);
    auto gmemUsed = allocator.assignGaddr(targetOuts.second, liveRange, true,
                                          sharedGmemOffset);
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
    updateAddressOfInPlaceOp(v_info, op_infos, neuron_alignment);
  }

  // TODO markGmemReusedOp
  // TODO crop concat pattern

  Module::setNeuronSize(module, sharedGmemSize);
  Module::setGmemPrivateSize(module, privateGmemSize);
  Module::updateModuleTypes(module);
  Module::setState(module, Module::State::TPU_ADDRESSED);
}

void CVAddressAssign::updateLiveRangeofPreOp(
    std::map<ValueInfo, OpElement> &op_infos, Operation *op, uint32_t end,
    std::map<Operation *, uint32_t> &ops_loc, MemType mem_type,
    int64_t alignment) {
  for (int i = 0; i < op->getNumOperands(); ++i) {
    auto operand = Module::getOperand(op, i);
    if (operand.getType().isa<mlir::NoneType>()) {
      continue;
    }
    auto preOp = operand.getDefiningOp();
    if (isa<top::WeightOp, top::NoneOp>(preOp)) {
      continue;
    }
    ValueInfo v_info(preOp, operand.cast<OpResult>().getResultNumber());
    if (isa<GenericCpuOp>(preOp)) {
      op_infos[v_info].mem_type = MEM_PRIVATE;
    }
    op_infos[v_info].live.start =
        std::min(ops_loc[preOp], op_infos[v_info].live.start);
    op_infos[v_info].live.end = std::max(end, op_infos[v_info].live.end);
    op_infos[v_info].live.out_index = v_info.index;
    if (0 == op_infos[v_info].live.tensor_size) {
      op_infos[v_info].live.tensor_size =
          getTensorGmemSize(preOp, v_info.index, alignment);
    }
    op_infos[v_info].mem_type = std::min(op_infos[v_info].mem_type, mem_type);
  }
}

void CVAddressAssign::updateLiveRangeOfInPlaceOp(
    std::map<ValueInfo, OpElement> &op_infos, Operation *op, uint32_t end,
    std::map<Operation *, uint32_t> &ops_loc, MemType mem_type,
    int64_t alignment) {
  if (auto concatOp = dyn_cast<tpu::ConcatOp>(op)) {
    // For ConcatN. To solve concat opt when axis = 0,
    // it need the operand should be continuous global memory.
    uint32_t tensor_size = getTensorGmemSize(op, 0, alignment);
    uint32_t max_end = end;
    uint32_t min_start = end;
    auto target_v = ValueInfo(0, 0);
    for (int i = 0; i < op->getNumOperands(); ++i) {
      auto operand = Module::getOperand(op, i);
      auto preOp = operand.getDefiningOp();
      ValueInfo v_info(preOp, operand.cast<OpResult>().getResultNumber());
      op_infos[v_info].live.start =
          std::min(ops_loc[preOp], op_infos[v_info].live.start);
      max_end = std::max(end, op_infos[v_info].live.end);
      op_infos[v_info].live.end = max_end;
      op_infos[v_info].live.out_index = v_info.index;
      op_infos[v_info].live.tensor_size = 0;
      op_infos[v_info].mem_type = std::min(op_infos[v_info].mem_type, mem_type);
      op_infos[v_info].need_alloc = false;
      if (op_infos[v_info].live.start < min_start) {
        target_v = v_info;
        min_start = op_infos[v_info].live.start;
      }
    }
    op_infos[target_v].live.end = max_end;
    op_infos[target_v].live.tensor_size = tensor_size;
    op_infos[target_v].need_alloc = true;
    op_infos[ValueInfo(op, 0)].target_v = target_v;
  } else {
    updateLiveRangeofPreOp(op_infos, op, end, ops_loc, mem_type, alignment);
  }
}

//  backward update
//  each step do
//  1. update cur op's (mem_type)
//  2. update pre op's (live.start live.end, mem_type, need_alloc)

void CVAddressAssign::updateLiveRange(Operation *op,
                                      std::map<Operation *, uint32_t> &ops_loc,
                                      std::map<ValueInfo, OpElement> &op_infos,
                                      std::vector<mlir::Value> &outputs,
                                      int64_t alignment) {
  if (isa<top::InputOp>(op)) {
    ValueInfo v_info(op, 0);
    op_infos[v_info].mem_type = MEM_IOMEM;
  } else if (isa<func::ReturnOp>(op)) {
    MemType mem_type = MEM_PRIVATE;
    auto func_op = dyn_cast<FuncOp>(op->getParentOp());
    assert(func_op);
    if (func_op.getName() == "main") {
      mem_type = MEM_IOMEM;
    }
    updateLiveRangeofPreOp(op_infos, op, ops_loc[op] + 1, ops_loc, mem_type,
                           alignment);
  } else if (Module::isOpInGroup(op)) {
  } else if (isInPlaceOp(op)) {
    ValueInfo cur_info(op, 0);
    assert(op_infos.find(cur_info) != op_infos.end());
    op_infos[cur_info].need_alloc = false;
    op_infos[cur_info].inplace = true;
    updateLiveRangeOfInPlaceOp(op_infos, op, op_infos[cur_info].live.end,
                               ops_loc, op_infos[cur_info].mem_type, alignment);
  } else if (op->getDialect()->getNamespace() == "tpu") {
    uint32_t max_end = 0;
    for (int i = 0; i < op->getNumResults(); ++i) {
      ValueInfo cur_info(op, i);
      assert(op_infos.find(cur_info) != op_infos.end());
    }
    updateLiveRangeofPreOp(op_infos, op, ops_loc[op] + 1, ops_loc, MEM_SHARED,
                           alignment);
  } else {
  }
}

void CVAddressAssign::updateAddressOfInPlaceOp(
    ValueInfo &v_info, std::map<ValueInfo, OpElement> &op_infos,
    int64_t alignment) {
  auto op = static_cast<Operation *>(v_info.op);
  if (auto concatOp = dyn_cast<tpu::ConcatOp>(op)) {
    int64_t base_addr = -1;
    ValueInfo cur_v(op, 0);
    auto target_v = op_infos[cur_v].target_v;
    base_addr = Module::getAddress(
        static_cast<Operation *>(target_v.op)->getResult(target_v.index));
    int64_t offset = 0;
    Module::setAddress(op->getResult(0), base_addr + offset);
    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      auto operand = Module::getOperand(op, i);
      auto opd = operand.getDefiningOp();
      if (opd == 0x0) {
        assert(0);
      }
      int this_index = operand.cast<OpResult>().getResultNumber();
      uint32_t tensor_size = getTensorGmemSize(opd, this_index, alignment);
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

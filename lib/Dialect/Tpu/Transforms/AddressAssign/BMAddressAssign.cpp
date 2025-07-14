//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "BMAddressAssign.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Backend/BM168x/MARS3.h"
#include "tpu_mlir/Backend/BM168x/SG2380.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/TPUNnvlcUtil.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "addressAssgin"

using namespace llvm;

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

bool valuesReturn(Value value) {
  for (auto op : value.getUsers()) {
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      return true;
    }
    if (BMAddressAssign::isInPlaceOp(op)) {
      for (auto v : op->getResults()) {
        if (valuesReturn(v)) {
          return true;
        }
      }
    }
  }
  return false;
}

static int64_t getIOLimit(ModuleOp m) {
  auto main = module::getMainFuncOp(m);
  int64_t limit = 0;
  std::vector<Value> io_v;
  main.walk([&](top::InputOp op) { io_v.push_back(op.getOutput()); });
  auto retOp = main.getBody().back().getTerminator();
  for (auto v : retOp->getOperands()) {
    io_v.push_back(v);
  }
  for (auto v : io_v) {
    auto l = align_up(module::getAddress(v) + module::getBytes(v),
                      BM168x::ALIGNMENT);
    if (l > limit) {
      limit = l;
    }
  }
  return limit;
}

std::set<ValueInfo> _8ChannelAssign(std::map<ValueInfo, TensorLive> &liveRange,
                                    bool reuse_addr) {
  if (!module::isBM1690Family())
    return {};

  std::set<ValueInfo> _8chOut;
  std::set<ValueInfo> _32chIn;
  std::map<ValueInfo, TensorLive> _8channelLiveRange;
  std::set<ValueInfo> AllSplitOpAndJoinOp;
  int64_t start_addr = 0;
  ValueInfo last_value;

  for (auto rit = liveRange.rbegin(); rit != liveRange.rend(); ++rit) {
    auto value = (*rit).first;
    auto live = (*rit).second;
    auto op = (Operation *)value.op;
    if (!isa<tpu::CoreParallelOp>(op)) {
      continue;
    }
    int op_splits = 0;
    for (auto &inner_op : op->getRegion(0).getOps()) {
      if (!isa<tpu::JoinOp>(inner_op))
        continue;
      op_splits = inner_op.getNumOperands();
      break;
    }

    auto result = op->getResult(0);
    ValueInfo new_value;
    int OnceFlag = 1;
    for (auto &use : result.getUses()) {
      if (isa<tpu::SplitOp>(use.getOwner())) {
        // this break is used to hack for tpu::ConvOp, when there are too few to
        // split into 8channels
        if (use.getOwner()->getNumResults() != op_splits) {
          break;
        }

        if (OnceFlag) {
          _8chOut.insert(value);
          op->walk([&](tpu::JoinOp join_op) {
            uint32_t per_size = Arch::get_gmem_bytes(join_op.getOperand(0));
            _8channelLiveRange.insert(std::pair<ValueInfo, TensorLive>(
                value, TensorLive(live.start, live.end, per_size)));
            AllSplitOpAndJoinOp.insert(value);
          });
          OnceFlag = 0;
        }

        int cur_id = -use.getOperandNumber() - 1;
        new_value = ValueInfo(use.getOwner(), cur_id);
        AllSplitOpAndJoinOp.insert(new_value);
      }
    }
  }

  auto getValues = [](std::map<ValueInfo, TensorLive> &valueMap) {
    std::vector<ValueInfo> values;
    values.reserve(valueMap.size());
    for (auto &[key, v] : valueMap)
      values.push_back(key);
    return std::move(values);
  };

  auto ops = getValues(_8channelLiveRange);
  std::map<ValueInfo, int64_t> _8ChannelMap;
  if (!ops.empty()) {
    // FitFirstAssign should make sure op's start liverange ascendingly
    GmemAllocator::sortOpByLiveStart(ops, _8channelLiveRange);
    GmemAllocator allocator(_8ChannelMap, BM168x::ALIGNMENT);
    auto _8channelUsed =
        allocator.assignGaddr(ops, _8channelLiveRange, reuse_addr, start_addr);
    LLVM_DEBUG(llvm::dbgs() << "_8channel Memory Each usage(without weight): "
                            << _8channelUsed / (1 << 20) << " MB\n");
  }

  for (auto &valueInfo : AllSplitOpAndJoinOp) {
    auto op = (Operation *)valueInfo.op;
    if (!isa<tpu::CoreParallelOp>(op))
      continue;
    int64_t offset;

    auto setCPAttr = [&op](auto &OpValue, auto &offset) {
      auto valueType =
          dyn_cast_or_null<mlir::RankedTensorType>(OpValue.getType());
      auto valueAttr = valueType.getEncoding();
      if (isa_and_present<tpu::CPInterleaveAttr>(valueAttr)) {
        return;
      }
      auto context = op->getContext();
      auto index_init = -1;
      auto ddrAttr = tpu::CPInterleaveAttr::get(context, index_init, offset, 0);
      auto ddrType = mlir::RankedTensorType::get(
          valueType.getShape(), valueType.getElementType(), ddrAttr);
      OpValue.setType(ddrType);
    };

    auto getOffset = [&](auto valueInfo) {
      if (_8ChannelMap.count(valueInfo) != 0) {
        offset = _8ChannelMap[valueInfo];
      } else {
        offset = -1;
      }
      return offset;
    };

    if (valueInfo.index >= 0) {
      auto result = op->getResult(0);
      offset = getOffset(valueInfo);
      setCPAttr(result, offset);
    } else {
      auto operand = op->getOperand(-valueInfo.index - 1);
      auto prev_op = op->getPrevNode();
      assert(isa<tpu::CoreParallelOp>(prev_op));
      assert(prev_op->getResult(0) == operand);
      offset = getOffset(ValueInfo(prev_op, -1));
      setCPAttr(operand, offset);
    }
  }
  return std::move(_8chOut);
}

static void fix_addr_for_io_tag(mlir::ModuleOp &m, int64_t start, int64_t limit,
                                int64_t offset) {
  for (auto func : m.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (isa<top::NoneOp, top::WeightOp, func::ReturnOp>(op)) {
        // do nothing
      } else {
        for (auto v : op->getResults()) {
          auto addr = module::getAddress(v);
          if (addr >= start && addr <= limit) {
            module::setAddress(v, addr + offset);
          }
        }
      }
    });
  }
}

static void fix_addr_for_io_alone(mlir::ModuleOp &m, int64_t start,
                                  int64_t io_limit, int64_t limit,
                                  int64_t io_offset, int64_t ctx_offset) {
  for (auto func : m.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (isa<top::NoneOp, top::WeightOp, func::ReturnOp>(op)) {
        // do nothing
      } else {
        for (auto v : op->getResults()) {
          auto addr = module::getAddress(v);
          if (addr >= start && addr < io_limit) {
            module::setAddress(v, addr + io_offset);
          } else if (addr >= io_limit && addr < limit) {
            module::setAddress(v, addr + ctx_offset);
          }
        }
      }
    });
  }
}

/** Set cmd-io-addr to new addresses. (The new addr spaces will NOT be allocated
 * by runtime, they are only used as FAKE addrs.) Inplace optimizations are also
 * kept as many as we can. */
static int64_t fix_addr_for_io_reloc(int64_t addr_limit, mlir::ModuleOp &m) {

  auto is_contain = [](const int64_t &a_start, const int64_t &a_end,
                       const int64_t &b_start, const int64_t &b_end) -> bool {
    return a_start <= b_start && a_end >= b_end; // a contains b => true.
  };
  auto get_addr_interval = [](Value v, int64_t &start, int64_t &end) -> void {
    start = module::getAddress(v);
    end = start + module::getBytes(v) - 1;
  };

  const int64_t alignment = BM168x::ALIGNMENT;
  // TODO: allow output reuse input addr. E.g. %output = tpu.Slice(%input), now
  // in-palce opt is not applied.
  std::vector<Value> input_values, output_values, io_values_must_fix;
  module::getInputsOutputs(m, input_values, output_values);
  io_values_must_fix.insert(io_values_must_fix.end(), input_values.begin(),
                            input_values.end());
  io_values_must_fix.insert(io_values_must_fix.end(), output_values.begin(),
                            output_values.end());

  for (auto io_var : io_values_must_fix) {
    int64_t io_start, io_end;
    get_addr_interval(io_var, io_start, io_end);
    int64_t addr_offset = addr_limit - io_start;
    module::setAddress(io_var, addr_limit);
    llvm::outs() << "[io_reloc] Fix IO addr for: " << module::getName(io_var)
                 << "\n";
    addr_limit += align_up(module::getBytes(io_var), alignment);

    // support some (but not all) in-place ops
    for (auto func : m.getOps<FuncOp>()) {
      bool dump_ = func.getName() != "main";
      func.walk([&](Operation *op) {
        if (isa<top::NoneOp, top::WeightOp, func::ReturnOp>(op)) {
          // do nothing
        } else {
          for (auto v : op->getResults()) {
            int64_t imm_start, imm_end;
            get_addr_interval(v, imm_start, imm_end);
            if (is_contain(io_start, io_end, imm_start, imm_end)) {
              module::setAddress(v, imm_start + addr_offset);
              if (dump_) {
                llvm::outs()
                    << "[io_reloc] Fix IO addr for: " << module::getName(v)
                    << "\n";
              }
            }
          }
        }
      });
    }
  }
  return addr_limit;
}

static void sort_ios(std::vector<Value> &ios) {
  std::sort(ios.begin(), ios.end(),
            [](const mlir::Value &a, const mlir::Value &b) {
              return module::getBytes(a) > module::getBytes(b);
            });
}

void BMAddressAssign::updateAddressByAddrMode(mlir::ModuleOp &m,
                                              int64_t start_addr,
                                              int64_t addr_limit) {
  if (module::isAddrMode(module::AddrMode::BASIC)) {
    module::setNeuronAddr(m, start_addr);
    module::setNeuronSize(m, addr_limit - start_addr);
    return;
  }
  if (module::isAddrMode(module::AddrMode::IO_RELOC)) {
    int64_t new_addr_limit = fix_addr_for_io_reloc(addr_limit, m);
    module::setNeuronAddr(m, start_addr);
    module::setNeuronSize(m, new_addr_limit - start_addr);
    module::updateModuleTypes();
    return;
  }
  auto io_limit = getIOLimit(m);
  if (module::isAddrMode(module::AddrMode::IO_TAG)) {
    // // assert(module::isBM1688());
    // std::vector<Value> ios;
    // module::getInputsOutputs(m, ios, ios);
    // // fix input and output address to IO_TAG
    // int io_index = 0;
    // int tag_max = 5;
    // if (ios.size() > tag_max) {
    //   // select IO with max 5 data size
    //   sort_ios(ios);
    //   for (io_index = 0; io_index < tag_max; io_index++) {
    //     module::setAddress(ios[io_index], BM168x::IO_ADDR[io_index]);
    //   }
    // } else {
    // for (auto &io : ios) {
    //   module::setAddress(io, BM168x::IO_ADDR[io_index++]);
    // }
    // }
    // fix other address
    module::setNeuronAddr(m, start_addr);
    module::setNeuronSize(m, addr_limit - start_addr);
    module::updateModuleTypes();
    return;
  }
  if (module::isAddrMode(module::AddrMode::IO_TAG_FUSE)) {
    assert(module::isBM1688());
    std::vector<Value> ins, outs;
    module::getInputsOutputs(m, ins, outs);
    // fix input and output address to IO_TAG
    int io_index = 0;
    int in_tag = 0, out_tag = 1;
    int64_t in_offset = 0, out_offset = 0;
    // fuse inputs onto in_tag
    for (io_index = 0; io_index < ins.size(); io_index++) {
      int64_t addr = BM168x::IO_ADDR[in_tag] + in_offset;
      module::setAddress(ins[io_index], addr);
      in_offset += module::getBytes(ins[io_index]);
    }
    // fuse outputs onto out_tag
    for (io_index = 0; io_index < outs.size(); io_index++) {
      int64_t addr = BM168x::IO_ADDR[out_tag] + out_offset;
      module::setAddress(outs[io_index], addr);
      out_offset += module::getBytes(outs[io_index]);
    }
    // fix other address
    module::setNeuronAddr(m, start_addr);
    module::setNeuronSize(m, addr_limit - start_addr);
    module::updateModuleTypes();
    return;
  }
  if (module::isAddrMode(module::AddrMode::IO_ALONE)) {
    if (module::isBM1684X()) {
      module::setIOAddr(m, start_addr);
      module::setIOSize(m, io_limit - start_addr);
      module::setNeuronAddr(m, io_limit);
      module::setNeuronSize(m, addr_limit - io_limit);
      return;
    }
    // move address to tag start
    int64_t io_start = 0x100000000ull;
    int64_t io_offset = io_start - start_addr;
    int64_t ctx_offset = start_addr - io_limit;
    fix_addr_for_io_alone(m, start_addr, io_limit, addr_limit, io_offset,
                          ctx_offset);
    module::setIOAddr(m, io_start);
    module::setIOSize(m, io_limit - start_addr);
    module::setNeuronAddr(m, start_addr);
    module::setNeuronSize(m, addr_limit - io_limit);
    module::updateModuleTypes();
    return;
  }
  llvm_unreachable("unknown addr_mode");
  return;
}

void BMAddressAssign::assignAfter(ModuleOp &m,
                                  std::vector<ValueInfo> &inplace_ops) {

  // Update ModuleTypes first to assure tensor address can be got on func level
  // then assignAfter can fix SplitOp IO addresses correctly
  // especially for the case: static subnet->dynamic subnet->static subnet(with
  // CoreParallel/SplitOp)
  module::updateModuleTypes();

  // step 0: assign inplace ops
  std::reverse(inplace_ops.begin(), inplace_ops.end());
  for (auto v_info : inplace_ops) {
    Operation *op = (Operation *)v_info.op;
    if (auto concatOp = dyn_cast<tpu::ConcatOp>(op)) {
      auto in0 = concatOp.getInputs()[0];
      in0 = module::getOriValue(in0);
      if (auto rop = dyn_cast<tpu::ReshapeOp>(in0.getDefiningOp())) {
        in0 = rop.getInput();
      }
      int64_t addr = module::getAddress(in0);
      if (addr == 0) {
        continue;
      }
      module::setAddress(concatOp.getOutput(), addr);
      int64_t offset = module::getBytes(in0);
      for (uint32_t i = 1; i < concatOp.getInputs().size(); i++) {
        auto input = concatOp.getInputs()[i];
        input = module::getOriValue(input);
        if (auto rop = dyn_cast<tpu::ReshapeOp>(input.getDefiningOp())) {
          module::setAddress(input, addr + offset);
          input = rop.getInput();
        }
        if (module::isAddrMode(module::AddrMode::IO_TAG) &&
            module::getAddress(input) >= BM168x::IO_ADDR[0]) {
          continue;
        }
        module::setAddress(input, addr + offset);
        offset += module::getBytes(input);
      }
    } else if (auto reshapeOp = dyn_cast<tpu::ReshapeOp>(op)) {
      auto addr = module::getAddress(reshapeOp.getInput());
      if (addr == 0) {
        addr = module::getAddress(module::getOriValue(reshapeOp.getOperand(0)));
      }
      if (addr == 0) {
        continue;
      }
      module::setAddress(reshapeOp.getOutput(), addr);
    } else if (auto identityOp = dyn_cast<tpu::IdentityOp>(op)) {
      for (auto it : llvm::enumerate(identityOp.getInput())) {
        auto addr = module::getAddress(module::getOriValue(it.value()));
        if (addr == 0) {
          continue;
        }
        module::setAddress(identityOp.getOutput()[it.index()], addr);
      }
    } else if (auto autoincOp = dyn_cast<tpu::AutoIncreaseOp>(op)) {
      auto addr = module::getAddress(module::getOriValue(autoincOp.getInput()));
      if (addr == 0) {
        continue;
      }
      module::setAddress(autoincOp.getOutput(), addr);
    } else if (auto sliceOp = dyn_cast<tpu::SliceOp>(op)) {
      auto addr = module::getAddress(module::getOriValue(sliceOp.getInput()));
      if (addr == 0) {
        continue;
      }
      auto p = sliceOp.parseParam();
      int axis;
      for (axis = 0; p.offset_4[axis] == 0 && axis < 4; axis++)
        ;
      size_t offset_bytes = 0;
      if (axis != 4) {
        auto _offset = p.offset_4[axis] < 0 ? p.offset_4[axis] + p.is_4[axis]
                                            : p.offset_4[axis];
        offset_bytes = _offset * module::getDtypeSize(sliceOp.getOutput());
        for (int i = axis + 1; i < 4; ++i) {
          offset_bytes *= p.is_4[i];
        }
      }
      module::setAddress(sliceOp.getOutput(), addr + offset_bytes);
    } else if (auto weight2activation_op =
                   dyn_cast<tpu::Weight2ActivationOp>(op)) {
      auto addr = module::getAddress(weight2activation_op.getInput());
      if (addr == 0) {
        continue;
      }
      module::setAddress(weight2activation_op.getOutput(), addr);
    } else {
      llvm_unreachable("set address of undefined inplace op!");
    }
  }
  // step 1: assign group ops
  for (auto func : m.getOps<FuncOp>()) {
    for (auto gOp : func.getOps<tpu::GroupOp>()) {
      auto &last_op = gOp.getBody().back().back();
      auto yield_op = dyn_cast<tpu::YieldOp>(last_op);
      assert(yield_op);
      int idx = 0;
      for (auto opd : yield_op.getOperands()) {
        auto addr = module::getAddress(gOp.getResult(idx));
        if (addr != 0) {
          module::setAddress(opd, addr);
        }
        idx++;
      }
    }
  }

  // step 2: populate groupParallel address to its regions.
  for (auto func : m.getOps<FuncOp>()) {
    for (auto groupParallelOp : func.getOps<tpu::GroupParallelOp>()) {
      for (auto [value, region] : llvm::zip(groupParallelOp.getResults(),
                                            groupParallelOp.getParallel())) {
        region.back().getTerminator()->getOperand(0).setType(value.getType());
      }
    }
  }
  // step 3: set parallel Op address
  auto If8channel = [](auto &checkOp, bool isSplitOp) {
    mlir::RankedTensorType valueType;
    if (isSplitOp) {
      valueType = dyn_cast_or_null<mlir::RankedTensorType>(
          ((tpu::SplitOp)checkOp).getOperand().getType());
    } else {
      valueType = dyn_cast_or_null<mlir::RankedTensorType>(
          checkOp.getResult(0).getType());
    }
    auto ddrAttr =
        dyn_cast_or_null<tpu::CPInterleaveAttr>(valueType.getEncoding());
    return ddrAttr;
  };
  for (auto func : m.getOps<FuncOp>()) {
    func.walk<WalkOrder::PreOrder>([&](tpu::CoreParallelOp parallelOp) {
      auto ifCPOut8ch = If8channel(parallelOp, false);
      for (auto &op : parallelOp.getRegion().getOps()) {
        llvm::TypeSwitch<Operation &>(op)
            .Case([&](tpu::SplitOp splitOp) {
              auto ifSplitOp8ch = If8channel(splitOp, true);
              if (!isa_and_present<tpu::CPInterleaveAttr>(ifSplitOp8ch)) {
                int64_t address = module::getAddress(splitOp->getOperand(0));
                if (address != 0) {
                  for (auto v : splitOp->getResults()) {
                    module::setAddress(v, address);
                    address += module::getBytes(v);
                  }
                }
              } else {
                for (auto [index, v] : llvm::enumerate(splitOp->getResults())) {
                  int64_t offset = ifSplitOp8ch.getOffset();
                  module::set8chAddress(v, index, offset, -1);
                }
              }
            })
            .Case([&](tpu::YieldOp yieldOp) {
              if (!isa_and_present<tpu::CPInterleaveAttr>(ifCPOut8ch)) {
                for (auto [joinOpValue, returnType] :
                     llvm::zip(yieldOp->getOperands(),
                               parallelOp->getResultTypes())) {
                  joinOpValue.setType(returnType);
                  if (!isa<tpu::JoinOp>(joinOpValue.getDefiningOp()))
                    continue;
                  int64_t address = module::getAddress(joinOpValue);
                  if (address == 0) {
                    continue;
                  }
                  for (auto v : joinOpValue.getDefiningOp()->getOperands()) {
                    if (v.getType().isa<NoneType>()) {
                      continue;
                    }
                    module::setAddress(v, address);
                    address += module::getBytes(v);
                  }
                }
              } else {
                for (auto [joinOpValue, returnType] :
                     llvm::zip(yieldOp->getOperands(),
                               parallelOp->getResultTypes())) {
                  joinOpValue.setType(returnType);
                  int64_t offset = ifCPOut8ch.getOffset();
                  for (auto [index, v] : llvm::enumerate(
                           joinOpValue.getDefiningOp()->getOperands())) {
                    module::set8chAddress(v, index, offset, -1);
                  }
                }
              }
            });
      }
    });
  }
}

void BMAddressAssign::assignL2SRAM(ModuleOp &m) {
  if (!module::isBM1690Family()) {
    return;
  }
  int64_t alignment = BM168x::ALIGNMENT;
  Builder builder(m.getContext());
  std::map<ValueInfo, TensorLive> liveRange;
  std::map<Operation *, uint32_t> ops_loc;
  std::vector<ValueInfo> common_ops;
  std::vector<ValueInfo> target_ops;
  std::vector<ValueInfo> inplace_ops;
  std::vector<Operation *> all_ops;
  uint32_t loc = 0;
  for (auto func : m.getOps<FuncOp>()) {
    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      ops_loc[op] = loc;
      ++loc;
      if (isa<FuncOp, top::NoneOp, top::WeightOp, top::InputOp, func::FuncOp,
              func::CallOp>(op) ||
          module::isOpInGroup(op)) {
        return;
      }
      if (module::isOpInCoreParallel(op) && !isa<tpu::BufferOp>(op)) {
        return;
      }
      all_ops.emplace_back(op);
    });
  }
  // update liverange from bottom to top.
  for (auto iter = all_ops.rbegin(); iter != all_ops.rend(); ++iter) {
    auto op = *iter;
    if (isa<ReturnOp, tpu::YieldOp>(op)) {
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

  for (auto &info : common_ops) {
    auto v = ((Operation *)info.op)->getResult(info.index);
    if (valuesReturn(v)) {
      continue;
    }
    target_ops.emplace_back(info);
  }
  int64_t l2memSize = BM168x::L2_SRAM_SIZE;
  auto core_num = module::getCoreNum();
  const int MAX_CORES = 8;
  l2memSize = (l2memSize / MAX_CORES) * core_num;

  int64_t start_addr = BM168x::L2_SRAM_START_ADDR;
  GmemAllocL2SRAM allocator(BM168x::ALIGNMENT, l2memSize);
  int64_t l2memUsed =
      allocator.assignGaddr(target_ops, liveRange, true, start_addr);
  if (l2memUsed > l2memSize) {
    llvm_unreachable("L2 mem allocate failed");
  }
  auto L2MemMap = allocator.getAddrMap();
  for (auto &[info, addr] : L2MemMap) {
    auto op = (Operation *)info.op;
    auto v = op->getResult(info.index);
    module::setAddress(v, addr);
  }
  assignAfter(m, inplace_ops);
}

void BMAddressAssign::assign(mlir::ModuleOp &m, bool reuse_addr) {
  int64_t alignment = BM168x::ALIGNMENT;
  int64_t start_addr = BM168x::COEFF_START_ADDR;
  Builder builder(m.getContext());
  // ========================= assign weight first ============================
  auto addr = start_addr;
  bool array_static[2] = {true, false};
  for (auto is_static : array_static) {
    if (!is_static) {
      module::setDynamicOffset(m, addr - start_addr);
    }
    for (auto func : m.getOps<FuncOp>()) {
      auto mode = getRunMode(func);
      if (mode == tpu::RunMode::UNKNOW) {
        mode = RunMode::TPU_STATIC;
      }
      if (is_static && mode != RunMode::TPU_STATIC) {
        continue;
      } else if (!is_static && mode == RunMode::TPU_STATIC) {
        continue;
      }
      func.walk([&](top::WeightOp op) { // static
        const auto out_value = op.getOutput();
        auto elm_bits =
            module::getStorageType(out_value).getIntOrFloatBitWidth();
        /// consider 4N/2N storage mode
        /// store_mode, align_num, dtype_size
        std::map<STORE_MODE_T, std::pair<int64_t, int32_t>> stmode_map = {
            {STORE_MODE_1N, {1l, elm_bits}},
            {STORE_MODE_2N, {2l, sizeof(int32_t) * 8}},
            {STORE_MODE_4N, {4l, sizeof(int32_t) * 8}},
        };
        auto stmode = STORE_MODE_1N;
        if (op.getStoreMode().has_value()) {
          stmode = llvm::StringSwitch<STORE_MODE_T>(op.getStoreModeAttr())
                       .Case("1N", STORE_MODE_1N)
                       .Case("2N", STORE_MODE_2N)
                       .Case("4N", STORE_MODE_4N)
                       .Default(STORE_MODE_1N);
        }
        assert((stmode == STORE_MODE_1N) ||
               (stmode == STORE_MODE_2N && elm_bits == 16) ||
               (stmode == STORE_MODE_4N && elm_bits == 8));

        module::setAddress(out_value, addr);
        int64_t n, c, h, w;
        module::getNCHW(out_value, n, c, h, w);
        int64_t bytes = ceiling_func(n, stmode_map.at(stmode).first) *
                        stmode_map.at(stmode).second * c * h * w;
        /// consider int4 storage
        bytes = ceiling_func(bytes, 8l);

        DEBUG_WITH_TYPE("gmem_allocator", {
          llvm::dbgs() << "; action = assignGaddr" << "; step = weight_static"
                       << "; start_addr = " << addr
                       << "; end_addr = " << addr + bytes
                       << "; live_start = " << 0
                       << "; live_end = " << 0x7FFFFFFF
                       << "; loc = " << module::getName(out_value).str()
                       << "\n";
        });

        addr = align_up(addr + bytes, alignment);
      });
    }
  }
  module::setCoeffAddr(m, start_addr);
  module::setCoeffSize(m, addr - start_addr);
  // ================= assign l2sram to activation =============================
  assignL2SRAM(m);
  // ================= assign ddr to activation ================================
  if (BM168x::SUPPORT_MEM_TAG) {
    addr = BM168x::CTX_START_ADDR;
  }
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
  for (auto func : m.getOps<FuncOp>()) {
    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      ops_loc[op] = loc;
      ++loc;
      if (isa<FuncOp, top::NoneOp, top::WeightOp>(op) ||
          module::isOpInGroup(op)) {
        return;
      }
      // The buffer Op will insert to parallel Op when needed.
      if (module::isOpInCoreParallel(op) && !isa<tpu::BufferOp>(op)) {
        return;
      }
      all_ops.emplace_back(op);
    });
  }
  // update liverange from bottom to top.
  for (auto iter = all_ops.rbegin(); iter != all_ops.rend(); ++iter) {
    auto op = *iter;
    if (isa<ReturnOp, tpu::YieldOp>(op)) {
      updateLiveRangeofBMOps(op, 0, ops_loc, liveRange, common_ops, inplace_ops,
                             alignment);
    }
    int n = op->getNumResults();
    for (int i = 0; i < n; i++) {
      auto v = op->getResult(i);
      if (module::isNone(v)) {
        continue;
      }
      updateLiveRangeofBMOps(op, i, ops_loc, liveRange, common_ops, inplace_ops,
                             alignment);
    }
  }

  // 8 channel
  // clang-format off
  // Tested: test_onnx.py with bm1690 f32 mode multicore.
  // Details: 8channel will only be activated when there are multiple
  //          continuous CoreParallelOp. the input of the 1st CoreParallelOp
  //          will be 32ch, the output of the final CoreParallelOp will
  //          be 32ch, the connections of these CoreParallelOp will be 8ch.
  // TODO: Support LayerGroupOp, single core and other mode<int8,f16,bf16>
  // clang-format on
  auto env_8ch = std::getenv("USING_8CH");
  if (env_8ch) {
    auto _8channelValueInfo = _8ChannelAssign(liveRange, reuse_addr);
    if (!_8channelValueInfo.empty()) {
      std::vector<ValueInfo> values;
      values.reserve(common_ops.size());
      for (auto v : common_ops) {
        if (_8channelValueInfo.count(v) == 0)
          values.push_back(v);
      }
      common_ops.swap(values);
    }
  }

  // 0. assign io-tag addrs before common_ops.
  if (module::isAddrMode(module::AddrMode::IO_TAG)) {
    auto erase_vinfo = [](std::vector<ValueInfo> &ops,
                          const ValueInfo &v_info) {
      auto iter = ops.begin();
      while (iter != ops.end()) {
        if (*iter == v_info) {
          iter = ops.erase(iter);
        } else {
          ++iter;
        }
      }
    };
    std::vector<Value> ios;
    module::getInputsOutputs(m, ios, ios);
    sort_ios(ios);
    int n_tags = ios.size() < 5 ? ios.size() : 5;
    for (int io_index = 0; io_index < n_tags; io_index++) {
      module::setAddress(ios[io_index], BM168x::IO_ADDR[io_index]);
      ValueInfo v_info(ios[io_index].getDefiningOp(),
                       ios[io_index].cast<OpResult>().getResultNumber());
      erase_vinfo(common_ops, v_info);
      erase_vinfo(inplace_ops, v_info);
      liveRange.erase(v_info);
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
    LLVM_DEBUG(llvm::dbgs() << "Global Memory usage(without weight): "
                            << gmemUsed / (1 << 20) << " MB\n");
  }

  for (auto &op_value : gaddrMap) {
    auto op = static_cast<Operation *>(op_value.first.op);
    module::setAddress(op->getResult(op_value.first.index), op_value.second);
  }

  // update io address by basic and io_tag
  if (!module::isAddrMode(module::AddrMode::IO_ALONE) &&
      !module::isAddrMode(module::AddrMode::IO_RELOC)) {
    updateAddressByAddrMode(m, start_addr, addr);
  }

  assignAfter(m, inplace_ops);

  module::updateModuleTypes();

  // update io address by io_alone
  if (module::isAddrMode(module::AddrMode::IO_ALONE) ||
      module::isAddrMode(module::AddrMode::IO_RELOC)) {
    updateAddressByAddrMode(m, start_addr, addr);
  }
}

static bool noNeedAddress(Value v) {
  if (module::isNone(v) || 0 != module::getAddress(v)) {
    return true;
  }
  return false;
}

void BMAddressAssign::updateLiveRangeofBMOps(
    Operation *op, int index, std::map<Operation *, uint32_t> &ops_loc,
    std::map<ValueInfo, TensorLive> &liveRange,
    std::vector<ValueInfo> &common_ops, std::vector<ValueInfo> &inplace_ops,
    int alignment) {
  auto updateOperandsLiveRange = [&](Operation *op, uint32_t endPosition) {
    DEBUG_WITH_TYPE("on_live_range", {
      llvm::dbgs() << "\n; action = updateOperandsLiveRange" << "; step = begin"
                   << "; op = " << module::getName(op)
                   << "; endPosition = " << endPosition << "\n";
    });
    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      auto operand = module::getOperand(op, i);
      auto opd = operand.getDefiningOp();
      DEBUG_WITH_TYPE("on_live_range", {
        llvm::dbgs() << "; action = updateOperandsLiveRange"
                     << "; step = opd_begin"
                     << "; opd_type = " << opd->getName()
                     << "; opd_loc = " << module::getName(operand)
                     << "; opd_index = " << i << "\n";
      });
      if (noNeedAddress(operand)) {
        DEBUG_WITH_TYPE("on_live_range", {
          llvm::dbgs() << "; action = updateOperandsLiveRange"
                       << "; step = opd_skip" << "\n";
        });
        continue;
      }
      ValueInfo v_info(opd, operand.cast<OpResult>().getResultNumber());
      DEBUG_WITH_TYPE("on_live_range", {
        llvm::dbgs() << "; action = updateOperandsLiveRange"
                     << "; step = opd_find"
                     << "; opd_loc = " << module::getName(operand)
                     << "; vinfo.index = " << v_info.index << "\n";
      });
      if (liveRange.find(v_info) != liveRange.end()) {
        // not first, update operand's liverange
        DEBUG_WITH_TYPE("on_live_range", {
          llvm::dbgs() << "; action = updateOperandsLiveRange"
                       << "; step = opd_first_meet_before"
                       << "; live_start = " << liveRange[v_info].start
                       << "; live_end = " << liveRange[v_info].end
                       << "; loc = " << module::getName(v_info.op)
                       << "; op = " << v_info.op->getName()
                       << "; position = " << ops_loc[opd] << "\n";
        });
        liveRange[v_info].start =
            std::min(liveRange[v_info].start, ops_loc[opd]);
        liveRange[v_info].end = std::max(liveRange[v_info].end, endPosition);
        DEBUG_WITH_TYPE("on_live_range", {
          llvm::dbgs() << "; action = update_live_range"
                       << "; step = opd_first_meet_after"
                       << "; live_start = " << liveRange[v_info].start
                       << "; live_end = " << liveRange[v_info].end
                       << "; loc = " << module::getName(v_info.op)
                       << "; op = " << v_info.op->getName()
                       << "; position = " << ops_loc[opd] << "\n";
        });
      } else {
        // first update the operand, set its start, end and tensor_size
        DEBUG_WITH_TYPE("on_live_range", {
          llvm::dbgs() << "; action = update_live_range"
                       << "; step = opd_second_meet_before"
                       << "; live_start = " << liveRange[v_info].start
                       << "; live_end = " << liveRange[v_info].end
                       << "; loc = " << module::getName(v_info.op)
                       << "; op = " << v_info.op->getName() << "\n";
        });
        liveRange[v_info].start = ops_loc[opd];
        liveRange[v_info].end = endPosition;
        liveRange[v_info].tensor_size =
            getTensorGmemSize(opd, v_info.index, alignment);

        DEBUG_WITH_TYPE("on_live_range", {
          llvm::dbgs() << "; action = update_live_range"
                       << "; step = opd_second_meet_after"
                       << "; live_start = " << liveRange[v_info].start
                       << "; live_end = " << liveRange[v_info].end
                       << "; loc = " << module::getName(v_info.op)
                       << "; op = " << v_info.op->getName()
                       << "; position = " << ops_loc[opd]
                       << "; tensor_size = " << liveRange[v_info].tensor_size
                       << "\n";
        });
      }

      if (isInPlaceOp(op)) {
        // if op is inplace op, operand live_end should be the same as op's
        ValueInfo op_info(op, 0);
        DEBUG_WITH_TYPE("live_range", {
          llvm::dbgs() << "; action = live_range"
                       << "; step = inplace_op_reset_before"
                       << "; live_start = " << liveRange[v_info].start
                       << "; live_end = " << liveRange[v_info].end
                       << "; loc = " << module::getName(v_info.op)
                       << "; inplace_op = " << module::getName(op)
                       << "; op = " << v_info.op->getName()
                       << "; op_info.live_range.end = "
                       << liveRange[op_info].end
                       << "; v_info.live_range.end = " << liveRange[v_info].end
                       << "\n";
        });
        liveRange[v_info].end =
            std::max(liveRange[op_info].end, liveRange[v_info].end);
        DEBUG_WITH_TYPE("live_range", {
          llvm::dbgs() << "; action = live_range"
                       << "; step = inplace_op_reset_after"
                       << "; loc = " << module::getName(operand)
                       << "; live_start = " << liveRange[v_info].start
                       << "; live_end = " << liveRange[v_info].end << "; "
                       << "\n";
        });
      }

      if (isa<top::InputOp>(opd) ||
          (isa<ReturnOp>(op) &&
           (module::isAddrMode(module::AddrMode::IO_ALONE) ||
            module::isAddrMode(module::AddrMode::IO_RELOC)))) {
        liveRange[v_info].start = 0;
        liveRange[v_info].end = 0xFFFFFFFF;
      }

      /* the operands of ops in prehead
         basic block will live forever */
      if (isa<tpu::LoopOp>(op)) {
        auto set_life_forerver = [&](ValueInfo &v_info) {
          if (liveRange.find(v_info) != liveRange.end()) {
            // not first, update operand's liverange
            liveRange[v_info].start = 0;
            liveRange[v_info].end = 0xFFFFFFFF;
          } else {
            // first update the operand, set its start, end and tensor_size
            liveRange[v_info].start = 0;
            liveRange[v_info].end = 0xFFFFFFFF;
            liveRange[v_info].tensor_size =
                getTensorGmemSize(opd, v_info.index, alignment);
          }
        };
        // loop mode: 6
        if (!isa<top::NoneOp>(
                module::getOriValue(op->getOperand(0)).getDefiningOp()) &&
            !isa<top::NoneOp>(
                module::getOriValue(op->getOperand(1)).getDefiningOp())) {
          /* Loop mode 6 : the prehead block IR as below:
             %0 = "tpu.Compare"(%arg0, %arg1) {mode = "Less"} :
                    (tensor<1xf32, 4295000064 : i64>, tensor<1xf32, 4294967296:
             i64>)
                    -> tensor<1xf32, 4294995968 : i64> loc(#loc11)
              %1 = "tpu.AutoIncrease"(%arg0) {const_val = 1.000000e+00 : f64}:
             (tensor<1xf32, 4295000064 : i64>)
                    -> tensor<1xf32, 4295000064 : i64> loc(#loc12)
              %2 = "tpu.Compare"(%0, %arg2) {mode = "And"} :
                    (tensor<1xf32, 4294995968 : i64>, tensor<1xf32, 4295012352:
             i64>)
                    -> tensor<1xf32, 4294991872 : i64> loc(#loc13)*/

          for (int i = 0; i < op->getNumOperands() - 2; i++) {
            // also set the life forerver
            auto operand = module::getOriValue(op->getOperand(i));
            auto opd = operand.getDefiningOp(); // other Op
            if (!noNeedAddress(operand)) {
              ValueInfo v_info(opd, operand.cast<OpResult>().getResultNumber());
              set_life_forerver(v_info);
            }
          }

          auto operand =
              module::getOriValue(op->getOperand(op->getNumOperands() - 2));
          auto opd = operand.getDefiningOp(); // AutoIncrease Op
          ValueInfo v_info(opd, operand.cast<OpResult>().getResultNumber());
          set_life_forerver(v_info);
          operand = module::getOriValue(opd->getOperand(0));
          opd = operand.getDefiningOp();
          if (!noNeedAddress(operand)) {
            ValueInfo v_info2(opd, operand.cast<OpResult>().getResultNumber());
            set_life_forerver(v_info2);
          }

          operand =
              module::getOriValue(op->getOperand(op->getNumOperands() - 1));
          opd = operand.getDefiningOp(); // Compare Op(And)
          ValueInfo v_info3(opd, operand.cast<OpResult>().getResultNumber());
          set_life_forerver(v_info3);

          auto dfs = [&](auto &&Me, Operation *opd) {
            if (!isa<tpu::CompareOp>(opd))
              return;

            for (int i = 0; i < opd->getNumOperands(); i++) {
              auto operand2 = module::getOriValue(opd->getOperand(i));
              auto opd2 = operand2.getDefiningOp();
              if (!noNeedAddress(operand2)) {
                ValueInfo v_info4(opd2,
                                  operand2.cast<OpResult>().getResultNumber());
                set_life_forerver(v_info4);
                Me(Me, opd2);
              }
            }
          };

          dfs(dfs, opd);
        }
      }

      DEBUG_WITH_TYPE("on_live_range", {
        llvm::dbgs() << "; action = updateOperandsLiveRange"
                     << "; step = opd_end" << "; opd_type = " << opd->getName()
                     << "; opd_loc = " << module::getName(operand)
                     << "; opd_index = " << i << "\n";
      });
    }
    DEBUG_WITH_TYPE("on_live_range", {
      llvm::dbgs() << "; action = updateOperandsLiveRange" << "; step = end"
                   << "; op = " << module::getName(op) << "\n";
    });
  };
  auto updateSOLOLiveRange = [&](Operation *op, ValueInfo v_info,
                                 uint32_t endPosition) {
    liveRange[v_info].start = ops_loc[op];
    liveRange[v_info].end = endPosition;
    liveRange[v_info].tensor_size =
        getTensorGmemSize(op, v_info.index, alignment);

    DEBUG_WITH_TYPE("live_range", {
      llvm::dbgs() << "; action = live_range" << "; step = update_solo"
                   << "; live_start = " << liveRange[v_info].start
                   << "; live_end = " << liveRange[v_info].end
                   << "; loc = " << module::getName(v_info.op)
                   << "; op = " << v_info.op->getName()
                   << "; tensor_size = " << liveRange[v_info].tensor_size
                   << "; index = " << v_info.index << "\n";
    });
  };
  ValueInfo v(op, index);
  uint32_t loc = ops_loc[op];
  uint32_t endPosition = loc + 1;
  if (auto nextOp = op->getNextNode()) {
    // This operation may have a region and the next operation can be treated
    // as the end of a scope.
    endPosition = ops_loc[nextOp];
  }
  if (isa<top::InputOp>(op)) {
    // liveRange.emplace_back(TensorLive(out, 0, 0xFFFFFFFF));
    // updateOperandsLiveRange(op, endPosition);
    if (module::getTrain()) {
      liveRange[v].start = 0;
      liveRange[v].end = 0x0FFFFFFF;
      liveRange[v].tensor_size = getTensorGmemSize(op, v.index, alignment);
    }
    common_ops.emplace_back(v);
    if (op->use_empty()) {
      liveRange[v].start = 0;
      liveRange[v].end = 0xFFFFFFFF;
      liveRange[v].tensor_size = getTensorGmemSize(op, v.index, alignment);
    }
    return;
  }
  if (isa<FuncOp, top::NoneOp, ReturnOp, top::WeightOp, func::CallOp,
          tpu::YieldOp>(op) ||
      module::isOpInGroup(op)) {
    /* for multi_subnet, the returnOp's live range increase if it connect to
       next subnet Todo: other complex case need to handle, such as it connect
       to next func's inner group op */
    // simple solution: don;t set the endlife if it connect to next subnet
    // currently. updateOperandsLiveRange(op, endPosition+2); Here,
    // updateLiveRange from the last op to the first op, no need to concern
    // it.
    updateOperandsLiveRange(op, endPosition);
    return;
  }

  if (isInPlaceOp(op)) {
    if (isa<tpu::ConcatOp>(op)) {
      updateOperandsLiveRange(op, endPosition);
      if (noNeedAddress(op->getResult(index))) {
        return;
      }
      uint32_t tensor_size = getTensorGmemSize(op, index, alignment);
      // liveRange[v] = TensorLive(index, loc, 0xFFFFFFFF, 0);
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
          DEBUG_WITH_TYPE("live_range", {
            llvm::dbgs() << "; action = live_range"
                         << "; step = concat_reshape_opd_reset"
                         << "; live_start = " << liveRange[pre_v].start
                         << "; live_end = " << liveRange[pre_v].end
                         << "; loc = " << module::getName(pre_v.op)
                         << "; op = " << pre_v.op->getName()
                         << "; tensor_size = " << liveRange[pre_v].tensor_size
                         << "; opd_index = " << i << "\n";
          });
        }
        ValueInfo pre_v(preOp, opd.cast<OpResult>().getResultNumber());
        liveRange[pre_v].start = concatLive[0];
        liveRange[pre_v].end = concatLive[1];
        if (i == 0) {
          liveRange[pre_v].tensor_size = tensor_size;
        } else {
          liveRange[pre_v].tensor_size = 0;
        }

        DEBUG_WITH_TYPE("live_range", {
          llvm::dbgs() << "; action = live_range" << "; step = inplace_concat"
                       << "; live_start = " << liveRange[pre_v].start
                       << "; live_end = " << liveRange[pre_v].end
                       << "; loc = " << module::getName(pre_v.op)
                       << "; op = " << pre_v.op->getName()
                       << "; tensor_size = " << liveRange[pre_v].tensor_size
                       << "; opd_index = " << i << "\n";
        });
      }
      inplace_ops.emplace_back(v);
    } else {
      uint32_t maxPosition = endPosition;
      findInPlaceOpMaxUsePosition(op, maxPosition, ops_loc);
      updateOperandsLiveRange(op, maxPosition);
      if (!noNeedAddress(op->getResult(index))) {
        inplace_ops.emplace_back(v);
      }
    }
    return;
  }
  if (isa_and_nonnull<tpu::GroupParallelOp>(op->getParentOp())) {
    // all the ops in parallel region have the liveRange the same as this
    // region.
    updateOperandsLiveRange(op, ops_loc[op->getParentOp()->getNextNode()]);
    if (!noNeedAddress(op->getResult(index))) {
      common_ops.emplace_back(v);
    }
    return;
  }
  if (isa_and_nonnull<tpu::CoreParallelOp>(op->getParentOp())) {
    auto upper = op->getParentOp()->getParentOp(); // nested liveRange
    if (isa_and_nonnull<tpu::GroupParallelOp>(upper))
      endPosition = ops_loc[upper->getNextNode()];
    else
      endPosition = ops_loc[op->getParentOp()->getNextNode()];
    updateSOLOLiveRange(op, v, endPosition);
    if (!noNeedAddress(op->getResult(index))) {
      common_ops.emplace_back(v);
    }
    return;
  }
  if (op->getDialect()->getNamespace() != "tpu" ||
      noNeedAddress(op->getResult(index))) {
    updateOperandsLiveRange(op, endPosition);
    return;
  }

  ValueInfo cur_info(op, index);
  if (liveRange.find(cur_info) == liveRange.end()) {
    updateSOLOLiveRange(op, cur_info, endPosition);
  } else {
    updateOperandsLiveRange(op, endPosition);
  }
  for (auto user : op->getUsers()) {
    if (isInPlaceOp(user)) {
      uint32_t maxPosition = endPosition;
      findInPlaceOpMaxUsePosition(user, maxPosition, ops_loc);
      updateOperandsLiveRange(op, maxPosition);
    }
  }
  common_ops.emplace_back(v);
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
  auto run_mode = tpu::getRunMode(op);
  if (module::isOpInBlock(op)) {
    return false;
  }
  if (auto ReshapeOp = dyn_cast<tpu::ReshapeOp>(op)) {
    if (Arch::ALIGN_4N &&
        module::getStorageType(ReshapeOp.getInput()).getIntOrFloatBitWidth() ==
            8) {
      int64_t in, ic, ih, iw, on, oc, oh, ow;
      module::getNCHW(ReshapeOp.getInput(), in, ic, ih, iw);
      module::getNCHW(ReshapeOp.getOutput(), on, oc, oh, ow);
      if (on != in) {
        return false;
      }
    }
    return true;
  } else if (auto sliceOp = dyn_cast<tpu::SliceOp>(op)) {
    if (run_mode == tpu::RunMode::TPU_DYNAMIC)
      return false;
    auto p = sliceOp.parseParam();
    return p.fusible;
  } else if (auto concatOp = dyn_cast<tpu::ConcatOp>(op)) {
    if (run_mode == tpu::RunMode::TPU_DYNAMIC)
      return false;
    return concatOp.getOnlyMerge();
  } else if (auto weight2activation_op =
                 dyn_cast<tpu::Weight2ActivationOp>(op)) {
    return true;
  } else if (isa<tpu::IdentityOp, tpu::AutoIncreaseOp>(op)) {
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
    auto operand = module::getOriValue(op->getOperand(i));
    auto pre_op = operand.getDefiningOp();
    int idx = operand.cast<OpResult>().getResultNumber();
    ValueInfo v_info(pre_op, idx);
    assert(liveRange.find(v_info) != liveRange.end());
    live[0] = std::min(liveRange[v_info].start, live[0]);
    live[1] = std::max(liveRange[v_info].end, live[1]);
  }
  return live;
}

uint32_t BMAddressAssign::getTensorGmemSize(Operation *op, int index,
                                            int64_t aligment_) {
  uint32_t size = Arch::get_gmem_bytes(op->getResult(index));

  // assign address for nnvlc
  bool do_compress = false;
  if (op != nullptr && isa<tpu::GroupOp>(op)) {
    auto yield_ = dyn_cast<GroupOp>(op).getOps<tpu::YieldOp>().begin();
    auto yieldop = *yield_;
    auto pre_op = yieldop->getOperand(index).getDefiningOp();
    if (isa<SliceMergeOp>(pre_op)) {
      pre_op = pre_op->getOperand(0).getDefiningOp();
    }
    auto storeop = dyn_cast<tpu::StoreOp>(pre_op);
    if (storeop->hasAttr("compress_info")) {
      auto cinfo_pre =
          storeop->getAttr("compress_info").cast<tpu::CompressAttr>();
      do_compress = cinfo_pre.getDoCompress();
    }
  } else if (op != nullptr && op->hasAttr("compress_info")) {
    auto cinfo = op->getAttr("compress_info").cast<tpu::CompressAttr>();
    do_compress = cinfo.getDoCompress();
  }
  if (do_compress) {
    std::vector<int64_t> shape = module::getShape(op->getResult(index));
    auto stype = module::getStorageType(op->getResult(index));
    shape_t ishape = {(int)shape[0], (int)shape[1], (int)shape[2],
                      (int)shape[3]};
    size_t max_meta_bytes = tpu_compress_RACU_max_meta_bytes(ishape);
    size_t max_racu_bytes = tpu_compress_RACU_max_racu_bytes(ishape, stype);
    size = std::max((size_t)size, align_up(max_meta_bytes, Arch::EU_BYTES) +
                                      align_up(max_racu_bytes, Arch::EU_BYTES));
  }

  // pad to aligment_
  if (size % aligment_) {
    size = size + aligment_ - (size % aligment_);
  }
  return size;
}

} // namespace tpu
} // namespace tpu_mlir

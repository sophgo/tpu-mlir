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
#include "tpu_mlir/Backend/BM168x/SG2380.h"
#include "tpu_mlir/Backend/CV18xx/CV184X.h"
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
  if (module::isAddrMode(module::AddrMode::IO_RELOC)) {
    int64_t new_addr_limit = fix_addr_for_io_reloc(addr_limit, m);
    module::setNeuronAddr(m, start_addr);
    module::setNeuronSize(m, new_addr_limit - start_addr);
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
  module::setNeuronAddr(m, start_addr);
  module::setNeuronSize(m, addr_limit - start_addr);
  return;
}

static void erase_vinfo(std::vector<ValueInfo> &ops, const ValueInfo &v_info) {
  for (auto iter = ops.begin(); iter != ops.end(); iter++) {
    if (*iter == v_info) {
      iter = ops.erase(iter);
      break;
    }
  }
}

static bool noNeedAddress(Value v) {
  if (module::isNone(v) || 0 != module::getAddress(v)) {
    return true;
  }
  return false;
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
  std::vector<ValueInfo> need_remove;
  // First assign concat ops
  for (auto v_info : inplace_ops) {
    Operation *op = (Operation *)v_info.op;
    if (auto concatOp = dyn_cast<tpu::ConcatOp>(op)) {
      if (0 != module::getAddress(concatOp.getOutput())) {
        need_remove.push_back(v_info);
        continue;
      }
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
      need_remove.push_back(v_info);
    }
  }
  // Then assign other inplace ops
  for (auto v_info : inplace_ops) {
    Operation *op = (Operation *)v_info.op;
    if (auto concatOp = dyn_cast<tpu::ConcatOp>(op)) {
      continue;
    } else if (auto reshapeOp = dyn_cast<tpu::ReshapeOp>(op)) {
      auto in_addr = module::getAddress(reshapeOp.getInput());
      auto out_addr = module::getAddress(reshapeOp.getOutput());
      if (in_addr == 0) {
        in_addr =
            module::getAddress(module::getOriValue(reshapeOp.getOperand(0)));
      }
      if (in_addr == 0 && out_addr == 0) {
        continue;
      }
      if (in_addr != 0 && out_addr != 0) {
        if (in_addr == out_addr) {
          need_remove.push_back(v_info);
        } else {
          UNREACHABLE_OP("ReshapeOp inplace address conflict!", op);
        }
        continue;
      }
      if (in_addr != 0) {
        module::setAddress(reshapeOp.getOutput(), in_addr);
      } else {
        module::setAddress(reshapeOp.getInput(), out_addr);
        auto v = module::getOriValue(reshapeOp.getOperand(0));
        module::setAddress(v, out_addr);
      }
      need_remove.push_back(v_info);
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
      for (axis = 0; axis < 4 && p.offset_4[axis] == 0; axis++)
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
      need_remove.push_back(v_info);
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
    for (auto gOp : func.getOps<tpu::GroupParallelOp>()) {
      for (auto [value, region] :
           llvm::zip(gOp.getResults(), gOp.getParallel())) {
        region.back().getTerminator()->getOperand(0).setType(value.getType());
        for (auto op : region.back().getOps<tpu::ReshapeOp>()) {
          auto addr = module::getAddress(op.getOutput());
          module::setAddress(op.getInput(), addr);
        }
      }
    }
  }
  // step 3: set parallel Op address
  for (auto func : m.getOps<FuncOp>()) {
    func.walk<WalkOrder::PreOrder>([&](tpu::CoreParallelOp parallelOp) {
      for (auto &op : parallelOp.getRegion().getOps()) {
        llvm::TypeSwitch<Operation &>(op)
            .Case([&](tpu::CoreSplitOp splitOp) {
              int64_t address = module::getAddress(splitOp->getOperand(0));
              if (address != 0) {
                for (auto v : splitOp->getResults()) {
                  module::setAddress(v, address);
                  address += module::getBytes(v);
                }
              }
            })
            .Case([&](tpu::YieldOp yieldOp) {
              for (auto [joinOpValue, returnType] : llvm::zip(
                       yieldOp->getOperands(), parallelOp->getResultTypes())) {
                joinOpValue.setType(returnType);
                if (!isa<tpu::CoreJoinOp>(joinOpValue.getDefiningOp()))
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
            });
      }
    });
  }
  // step 4: remove assigned inplace ops from inplace_ops
  for (auto &v_info : need_remove) {
    erase_vinfo(inplace_ops, v_info);
  }
  std::reverse(inplace_ops.begin(), inplace_ops.end());
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

static inline std::vector<std::pair<int, int>> string2pair(std::string slist) {
  // slist: "0:0,1:2,4:4"
  // outpair: [0,0] [1,2] [4,4]
  std::vector<std::pair<int, int>> outpair;
  std::string idx_str = "";
  int first = 0;
  int second = 0;
  for (auto s : slist) {
    if (s == ':') {
      first = atoi(idx_str.c_str());
      idx_str = "";
    } else if (s == ',') {
      second = atoi(idx_str.c_str());
      idx_str = "";
      outpair.emplace_back(first, second);
    } else if (s == ' ') {
      continue;
    } else {
      idx_str += s;
    }
  }
  if (idx_str.size()) {
    second = atoi(idx_str.c_str());
    outpair.emplace_back(first, second);
  }
  return outpair;
}

// To handle cases that inplace ops share the same address with ios
static void inplace_addr_update(std::vector<ValueInfo> &inplace_ops,
                                const ValueInfo &v_info) {
  Operation *v_op = (Operation *)v_info.op;
  Value v_result = v_op->getResult(v_info.index);
  int64_t v_addr = module::getAddress(v_result);

  for (auto &inplace_info : inplace_ops) {
    Operation *inplace_op = (Operation *)inplace_info.op;
    Value inplace_input = inplace_op->getOperand(0);
    Value inplace_output = inplace_op->getResult(inplace_info.index);

    // case1：current io is inplace op
    if (inplace_op == v_op) {
      module::setAddress(inplace_input, v_addr);
    }
    // case2：current io is input of inplace op
    else if (inplace_input == v_result) {
      module::setAddress(inplace_output, v_addr);
    }
  }
}

void BMAddressAssign::assignIOByAddrMode(
    ModuleOp &m, std::map<ValueInfo, TensorLive> &liveRange,
    std::vector<ValueInfo> &inplace_ops, std::vector<ValueInfo> &common_ops,
    int64_t &start_addr) {
  if (module::isAddrMode(module::AddrMode::IO_TAG)) {
    std::vector<Value> ios;
    module::getInputsOutputs(m, ios, ios);
    sort_ios(ios);
    int n_tags = ios.size() < 5 ? ios.size() : 5;
    for (int io_index = 0; io_index < n_tags; io_index++) {
      module::setAddress(ios[io_index], BM168x::IO_ADDR[io_index]);
      ValueInfo v_info(ios[io_index].getDefiningOp(),
                       ios[io_index].cast<OpResult>().getResultNumber());
      inplace_addr_update(inplace_ops, v_info);
      erase_vinfo(common_ops, v_info);
      erase_vinfo(inplace_ops, v_info);
      liveRange.erase(v_info);
    }
    return;
  }
  if (module::isAddrMode(module::AddrMode::IO_ALONE)) {
    int64_t io_start = start_addr;
    if (BM168x::SUPPORT_MEM_TAG) {
      io_start = BM168x::IO_START_ADDR;
    }
    std::vector<Value> ios;
    module::getInputsOutputs(m, ios, ios);
    auto addr = io_start;
    for (auto &v : ios) {
      auto v_info =
          ValueInfo(v.getDefiningOp(), v.cast<OpResult>().getResultNumber());
      auto bytes = liveRange[v_info].tensor_size;
      if (bytes == 0) {
        continue;
      }
      module::setAddress(v, addr);
      addr += bytes;
    }
    assignAfter(m, inplace_ops);
    for (auto &v : ios) {
      auto v_info =
          ValueInfo(v.getDefiningOp(), v.cast<OpResult>().getResultNumber());
      erase_vinfo(common_ops, v_info);
      erase_vinfo(inplace_ops, v_info);
      liveRange.erase(v_info);
      if (module::getAddress(v) == 0) {
        UNREACHABLE_OP("IO address assign failed", v_info.op);
      }
    }
    auto io_size = addr - io_start;
    module::setIOAddr(m, io_start);
    module::setIOSize(m, io_size);
    if (!BM168x::SUPPORT_MEM_TAG) {
      start_addr = addr;
    }
    return;
  }
}

void BMAddressAssign::assign(mlir::ModuleOp &m, bool reuse_addr,
                             std::string same_addr) {
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
        /// for safety of int4 MatMul Weight.
        if (stmode_map.at(stmode).second == 4 &&
            module::IsRightMat(out_value)) {
          int64_t n2, c2, h2, w2;
          module::getNCHW(out_value, n2, c2, h2, w2, /*left_align=*/false);
          int64_t bytes_2 = ceiling_func(n2, stmode_map.at(stmode).first) * c2 *
                            h2 *
                            ceiling_func(w2 * stmode_map.at(stmode).second, 8l);
          bytes = std::max(bytes, bytes_2);
        }

        DEBUG_WITH_TYPE("gmem_allocator", {
          llvm::dbgs() << "; action = assignGaddr"
                       << "; step = weight_static"
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

  // 0. assign io-tag/io-alone addrs before common_ops.
  assignIOByAddrMode(m, liveRange, inplace_ops, common_ops, start_addr);
  addr = start_addr;

  // 1.assign common_ops
  // key: the operation pointer + output index, convert the result to type
  // int64_t
  // clear ops that no need to assign address
  std::vector<ValueInfo> remove_ops;
  for (auto &info : common_ops) {
    auto v = ((Operation *)info.op)->getResult(info.index);
    if (noNeedAddress(v)) {
      remove_ops.emplace_back(info);
    }
  }
  for (auto &info : remove_ops) {
    erase_vinfo(common_ops, info);
    liveRange.erase(info);
  }
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

  std::vector<ValueInfo> part_inplace_ops;
  for (auto &op_value : gaddrMap) {
    auto op = static_cast<Operation *>(op_value.first.op);
    module::setAddress(op->getResult(op_value.first.index), op_value.second);
    auto result_index = getInplaceOperandIndex(op, op_value.first.index);
    if (result_index >= 0) {
      part_inplace_ops.emplace_back(op_value.first);
    }
  }

  // update io address by basic and io_tag
  if (!module::isAddrMode(module::AddrMode::IO_RELOC)) {
    updateAddressByAddrMode(m, start_addr, addr);
  }

  assignAfter(m, inplace_ops);

  for (auto value_info : part_inplace_ops) {
    auto op = static_cast<Operation *>(value_info.op);
    module::setAddress(op->getResult(value_info.index),
                       module::getAddress(op->getOperand(
                           getInplaceOperandIndex(op, value_info.index))));
  }
  module::updateModuleTypes();

  // update io address by io reloc
  if (module::isAddrMode(module::AddrMode::IO_RELOC)) {
    updateAddressByAddrMode(m, start_addr, addr);
  }

  // set the same address pairs
  if (!same_addr.empty()) {
    std::vector<std::pair<int, int>> same_idx = string2pair(same_addr);
    for (auto &pair : same_idx) {
      std::vector<Value> ins, outs;
      module::getInputsOutputs(m, ins, outs);
      module::setAddress(outs[pair.second],
                         module::getAddress(ins[pair.first]));
    }
    module::updateModuleTypes();
  }
}

void BMAddressAssign::updateLiveRangeofBMOps(
    Operation *op, int index, std::map<Operation *, uint32_t> &ops_loc,
    std::map<ValueInfo, TensorLive> &liveRange,
    std::vector<ValueInfo> &common_ops, std::vector<ValueInfo> &inplace_ops,
    int alignment) {
  auto updateOperandsLiveRange = [&](Operation *op, uint32_t endPosition) {
    DEBUG_WITH_TYPE("on_live_range", {
      llvm::dbgs() << "\n; action = updateOperandsLiveRange"
                   << "; step = begin"
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
                       << "; step = opd_skip"
                       << "\n";
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

      auto out_index = getInplaceResultIndex(op, i);
      if (out_index >= 0) {
        ValueInfo op_info(op, out_index);
        liveRange[op_info].tensor_size = 0; // inplace result tensor size is 0
        DEBUG_WITH_TYPE("on_live_range", {
          llvm::dbgs() << "; action = update_live_range"
                       << "; step = opd_second_meet_after"
                       << "; live_start = " << liveRange[op_info].start
                       << "; live_end = " << liveRange[op_info].end
                       << "; loc = "
                       << module::getName(op->getResult(out_index))
                       << "; op = " << op->getName()
                       << "; tensor_size = " << liveRange[op_info].tensor_size
                       << "\n";
        });
        liveRange[v_info].end =
            std::max(liveRange[op_info].end, liveRange[v_info].end);

        DEBUG_WITH_TYPE("on_live_range", {
          llvm::dbgs() << "; action = live_range"
                       << "; step = inplace_op_reset_after"
                       << "; loc = " << module::getName(operand)
                       << "; live_start = " << liveRange[v_info].start
                       << "; live_end = " << liveRange[v_info].end
                       << "; tensor_size = " << liveRange[v_info].tensor_size
                       << "\n";
        });
      }

      if ((isa<top::InputOp>(opd) &&
           !module::isAddrMode(module::AddrMode::IN_REUSE)) ||
          (isa<ReturnOp>(op) &&
           module::isAddrMode(module::AddrMode::IO_RELOC))) {
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
                     << "; step = opd_end"
                     << "; opd_type = " << opd->getName()
                     << "; opd_loc = " << module::getName(operand)
                     << "; opd_index = " << i << "\n";
      });
    }
    DEBUG_WITH_TYPE("on_live_range", {
      llvm::dbgs() << "; action = updateOperandsLiveRange"
                   << "; step = end"
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
      llvm::dbgs() << "; action = live_range"
                   << "; step = update_solo"
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
          llvm::dbgs() << "; action = live_range"
                       << "; step = inplace_concat"
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

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
#include "tpu_mlir/Support/MathUtils.h"

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

void BMAddressAssign::assign(mlir::ModuleOp &m, bool reuse_addr) {
  int64_t alignment = BM168x::ALIGNMENT;
  int64_t start_addr = BM168x::COEFF_START_ADDR;
  Builder builder(m.getContext());
  // assign weight first
  auto addr = start_addr;
  for (auto func : m.getOps<FuncOp>()) {
    func.walk([&](top::WeightOp op) {
      const auto out_value = op.getOutput();
      auto elm_bits = module::getStorageType(out_value).getIntOrFloatBitWidth();
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
      addr = align_up(addr + bytes, alignment);
    });
  }
  module::setCoeffAddr(m, start_addr);
  module::setCoeffSize(m, addr - start_addr);

  // assign activation
  if (module::isBM1686() || module::isSG2260Family()) {
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
      if (module::isOpInParallel(op) && !isa<tpu::BufferOp>(op)) {
        return;
      };
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
      if (addr == 0) {
        addr = module::getAddress(module::getOriValue(reshapeOp.getOperand(0)));
      }
      module::setAddress(reshapeOp.getOutput(), addr);
    } else if (auto identityOp = dyn_cast<tpu::IdentityOp>(op)) {
      for (auto it : llvm::enumerate(identityOp.getInput())) {
        auto addr = module::getAddress(module::getOriValue(it.value()));
        module::setAddress(identityOp.getOutput()[it.index()], addr);
      }
    } else if (auto autoincOp = dyn_cast<tpu::AutoIncreaseOp>(op)) {
      auto addr = module::getAddress(module::getOriValue(autoincOp.getInput()));
      module::setAddress(autoincOp.getOutput(), addr);
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
    } else if (auto weight2activation_op =
                   dyn_cast<tpu::Weight2ActivationOp>(op)) {
      module::setAddress(weight2activation_op.getOutput(),
                         module::getAddress(weight2activation_op.getInput()));
    } else {
      llvm_unreachable("set address of undefined inplace op!");
    }
  }

  // 3. set parallel Op address
  for (auto func : m.getOps<FuncOp>()) {
    func.walk<WalkOrder::PreOrder>([&](tpu::ParallelOp parallelOp) {
      for (auto &op : parallelOp.getRegion().getOps()) {
        llvm::TypeSwitch<Operation &>(op)
            .Case([](tpu::SplitOp splitOp) {
              int64_t address = module::getAddress(splitOp->getOperand(0));
              for (auto v : splitOp->getResults()) {
                module::setAddress(v, address);
                address += module::getBytes(v);
              }
            })
            .Case([&](tpu::YieldOp yieldOp) {
              for (auto joinOp_withType : llvm::zip(
                       yieldOp->getOperands(), parallelOp->getResultTypes())) {
                auto joinOpValue = std::get<0>(joinOp_withType);
                joinOpValue.setType(parallelOp->getResultTypes()[0]);
                int64_t address = module::getAddress(joinOpValue);
                for (auto v : joinOpValue.getDefiningOp()->getOperands()) {
                  module::setAddress(v, address);
                  if (isa<tpu::GroupOp>(v.getDefiningOp())) {
                    group_ops.push_back(ValueInfo(v.getDefiningOp(), 0));
                  } else {
                    address += module::getBytes(v);
                  }
                }
              }
            });
      }
    });
  }

  // 4.set group op address
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

  // 4. set parallel Op address
  for (auto func : m.getOps<FuncOp>()) {
    func.walk<WalkOrder::PreOrder>([&](tpu::ParallelOp op) {
      auto splitOp = &op.getBody().front().front();
      int64_t address = module::getAddress(splitOp->getOperand(0));
      for (auto v : splitOp->getResults()) {
        module::setAddress(v, address);
        address += module::getBytes(v);
      }
      auto yieldOp = &op.getBody().front().back();
      for (auto joinOp_withType :
           llvm::zip(yieldOp->getOperands(), op->getResultTypes())) {
        auto joinOpValue = std::get<0>(joinOp_withType);
        joinOpValue.setType(op.getResult(0).getType());
        address = module::getAddress(joinOpValue);
        for (auto v : joinOpValue.getDefiningOp()->getOperands()) {
          module::setAddress(v, address);
          address += module::getBytes(v);
        }
      }
    });
  }

  module::setNeuronAddr(m, start_addr);
  module::setNeuronSize(m, addr - start_addr);
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

      if (isInPlaceOp(op)) {
        ValueInfo op_info(op, 0);
        liveRange[v_info].end =
            std::max(liveRange[op_info].end, liveRange[v_info].end);
      }

      if (isa<top::InputOp>(opd)) {
        liveRange[v_info].start = 0;
        liveRange[v_info].end = 0xFFFFFFFF;
      }

      /* the operands of ops in prehead
         basic block will live forever */
      if (isa<tpu::LoopOp>(op)) {
        auto set_life_forerver = [&] (ValueInfo &v_info) {
          if (liveRange.find(v_info) != liveRange.end()) {
            // not first, update operand's liverange
            liveRange[v_info].start = 0;
            liveRange[v_info].end = 0xFFFFFFFF;
            liveRange[v_info].out_index = v_info.index;
          } else {
            // first update the operand, set its start, end, out_index and
            // tensor_size
            liveRange[v_info].start = 0;
            liveRange[v_info].end = 0xFFFFFFFF;
            liveRange[v_info].out_index = v_info.index;
            liveRange[v_info].tensor_size =
                getTensorGmemSize(opd, v_info.index, alignment);
          }
        };
        //loop mode: 6
        if (!isa<top::NoneOp>(module::getOriValue(op->getOperand(0)).getDefiningOp())
            && !isa<top::NoneOp>(module::getOriValue(op->getOperand(1)).getDefiningOp())) {
          /* Loop mode 6 : the prehead block IR as below:
             %0 = "tpu.Compare"(%arg0, %arg1) {mode = "Less"} :
                    (tensor<1xf32, 4295000064 : i64>, tensor<1xf32, 4294967296 : i64>)
                    -> tensor<1xf32, 4294995968 : i64> loc(#loc11)
              %1 = "tpu.AutoIncrease"(%arg0) {const_val = 1.000000e+00 : f64} :
                    (tensor<1xf32, 4295000064 : i64>)
                    -> tensor<1xf32, 4295000064 : i64> loc(#loc12)
              %2 = "tpu.Compare"(%0, %arg2) {mode = "And"} :
                    (tensor<1xf32, 4294995968 : i64>, tensor<1xf32, 4295012352 : i64>)
                    -> tensor<1xf32, 4294991872 : i64> loc(#loc13)*/

          for (int i = 0; i < op->getNumOperands() - 2; i++) {
            //also set the life forerver
            auto operand = module::getOriValue(op->getOperand(i));
            auto opd = operand.getDefiningOp(); //other Op
            if (!isa<top::WeightOp, top::NoneOp>(opd)) {
              ValueInfo v_info(opd, operand.cast<OpResult>().getResultNumber());
              set_life_forerver(v_info);
            }
          }

          auto operand = module::getOriValue(op->getOperand
                                (op->getNumOperands() - 2));
          auto opd = operand.getDefiningOp(); //AutoIncrease Op
          ValueInfo v_info(opd, operand.cast<OpResult>().getResultNumber());
          set_life_forerver(v_info);
          operand = module::getOriValue(opd->getOperand(0));
          opd = operand.getDefiningOp();
          if (!isa<top::WeightOp, top::NoneOp>(opd)) {
            ValueInfo v_info2(opd, operand.cast<OpResult>().getResultNumber());
            set_life_forerver(v_info2);
          }

          operand = module::getOriValue(op->getOperand
                                (op->getNumOperands() - 1));
          opd = operand.getDefiningOp(); //Compare Op(And)
          ValueInfo v_info3(opd, operand.cast<OpResult>().getResultNumber());
          set_life_forerver(v_info3);

          auto dfs = [&] (auto &&Me, Operation *opd) {
            if (!isa<tpu::CompareOp>(opd))
              return;

            for (int i = 0; i < opd->getNumOperands(); i++) {
              auto operand2 = module::getOriValue(opd->getOperand(i));
              auto opd2 = operand2.getDefiningOp();
              if (!isa<top::WeightOp, top::NoneOp>(opd2)) {
                ValueInfo v_info4(opd2, operand2.cast<OpResult>().getResultNumber());
                set_life_forerver(v_info4);
                Me(Me, opd2);
              }
            }
          };

          dfs(dfs, opd);
        }
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
  auto updateAsyncLiveRange = [&](Operation *op, ValueInfo v_info) {
    // all the ops in parallel region have the liveRange the same as this
    // region.
    liveRange[v_info].start = ops_loc[op];
    liveRange[v_info].end = ops_loc[op->getParentOp()->getNextNode()];
    liveRange[v_info].out_index = index;
    liveRange[v_info].tensor_size = getTensorGmemSize(op, index, alignment);
  };
  ValueInfo v(op, index);
  uint32_t loc = ops_loc[op];
  uint32_t endPosition = loc + 1;
  if (auto nextOp = op->getNextNode()) {
    // This operation may have a region and the next operation can be treated as
    // the end of a scope.
    endPosition = ops_loc[nextOp];
  }
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
    // updateLiveRange from the last op to the first op, no need to concern
    // it.
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
  } else if (module::isOpInParallel(op)) {
    updateAsyncLiveRange(op, v);
    common_ops.emplace_back(v);
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
    auto p = sliceOp.parseParam();
    return p.fusible;
  } else if (auto concatOp = dyn_cast<tpu::ConcatOp>(op)) {
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

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "CVAddressAssign.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "llvm/Support/MD5.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

std::string CVAddressAssign::calcMD5(std::vector<uint8_t> &data) {
  auto md5 = llvm::MD5::hash(data);
  SmallString<32> res;
  MD5::stringifyResult(md5, res);
  return std::string(res);
}

bool CVAddressAssign::loadAddressMapping(
    std::string &mapFileName,
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> &addrMapping) {
  auto stream =
      std::make_unique<std::fstream>(mapFileName.c_str(), std::fstream::in);
  if (!stream->is_open()) {
    return false;
  }

  char buf[512];
  while (!stream->eof()) {
    memset(buf, 0, sizeof(buf));
    stream->getline(buf, sizeof(buf));
    StringRef str(buf);
    if (str.empty()) {
      continue;
    }
    SmallVector<StringRef, 4> fields;
    str.split(fields, ',', -1, true);
    auto pos = fields[1].str();
    auto md5 = fields[2].str();
    auto length = fields[3].str();
    addrMapping[md5].first = std::stol(pos, nullptr, 16);
    addrMapping[md5].second = std::stol(length, nullptr, 10);
  }
  return true;
}

void CVAddressAssign::checkIfFileGood(std::string &fileName,
                                      std::unique_ptr<std::fstream> &stream) {
  if (!stream->is_open()) {
    llvm::errs() << "cannot open output file '" + fileName + "\n";
    assert(0);
  }
}

void CVAddressAssign::assign_weight_addr(mlir::ModuleOp &m, bool merge_weight,
                                         bool compress_weight,
                                         std::string &weight_map_file) {
  if (merge_weight) {
    compress_weight = false;
  }
  int64_t start_addr = (uint64_t)1 << 40;
  int64_t start_offset = 0;
  int64_t weight_alignment = 16;
  std::unordered_map<std::string, std::vector<top::WeightOp>> weight_md5_map;
  std::unordered_map<std::string, std::pair<int64_t, int64_t>> addrMapping;
  auto flags = std::fstream::out;
  assert(!weight_map_file.empty());
  if (merge_weight) {
    // load address from pre cvimodel
    if (loadAddressMapping(weight_map_file, addrMapping)) {
      flags = flags | std::fstream::app;
    }
    if (!addrMapping.empty()) {
      // find start addr according to pre cvimodel
      using item_type = std::pair<std::string, std::pair<int64_t, int64_t>>;
      auto iter = std::max_element(addrMapping.begin(), addrMapping.end(),
                                   [](item_type &&lhs, item_type &&rhs) {
                                     return lhs.second.first < rhs.second.first;
                                   });
      start_offset =
          align_up(iter->second.first + iter->second.second, weight_alignment);
    }
  }
  auto weightMapFile = std::make_unique<std::fstream>(weight_map_file, flags);
  checkIfFileGood(weight_map_file, weightMapFile);

  for (auto func : m.getOps<FuncOp>()) {
    func.walk([&](top::WeightOp op) {
      auto weight_data = op.read_as_byte();
      std::string md5 = calcMD5(*weight_data);
      weight_md5_map[md5].emplace_back(op);
      // set compress weight
      if (compress_weight && op.getResult().hasOneUse()) {
        auto nextOp = (*op.getResult().getUses().begin()).getOwner();
        if (isa<tpu::Conv2DOp>(nextOp) || isa<tpu::MatMulOp>(nextOp)) {
          Builder builder(m.getContext());
          op.setDoCompressAttr(builder.getBoolAttr(true));
        }
      }
    });
  }

  auto addr = start_offset;
  for (auto &pair : weight_md5_map) {
    int64_t offset = addr;
    int64_t bytes = module::getBytes(pair.second[0].getOutput());
    auto iter_redundant = addrMapping.find(pair.first);
    if (iter_redundant != addrMapping.end()) {
      offset = iter_redundant->second.first;
      assert(bytes == iter_redundant->second.second);
    } else {
      std::string s;
      llvm::raw_string_ostream os(s);
      os << module::getName(pair.second[0].getOutput()).str() << ","
         << llvm::format_hex(offset, 10) << "," << pair.first << "," << bytes
         << "\n";
      weightMapFile->write(os.str().c_str(), os.str().size());
      addr = align_up(addr + bytes, weight_alignment);
    }
    for (auto &op : pair.second) {
      module::setAddress(op.getOutput(), offset + start_addr);
    }
    if (pair.second.size() > 1 || iter_redundant != addrMapping.end()) {
      // redundant weight
      for (auto &weight_op : pair.second) {
        weight_op.removeDoCompressAttr();
      }
    }
  }
  module::setCoeffAddr(m, start_addr);
  module::setCoeffSize(m, addr);
}

void CVAddressAssign::assign(mlir::ModuleOp &m, bool reuse_addr,
                             bool merge_weight, bool compress_weight,
                             std::string &weight_map_file) {
  // assign weight first
  assign_weight_addr(m, merge_weight, compress_weight, weight_map_file);

  int64_t neuron_alignment = 16;
  // key: the operation pointer & output index
  std::map<Operation *, uint32_t> ops_loc;
  std::map<ValueInfo, TensorLive> liveRange;
  std::map<ValueInfo, OpElement> op_infos;
  std::vector<Operation *> ops;
  std::vector<ValueInfo> inplace_ops;
  std::map<std::string, std::vector<ValueInfo>> shared_outs_regions;
  std::vector<ValueInfo> private_outs;
  std::vector<ValueInfo> io_outs;
  std::vector<Operation *> group_ops;

  // assign activation
  uint32_t loc = 0;
  for (auto func : m.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      ops_loc[op] = loc;
      ++loc;
      if ((module::isTpuOp(op) && !module::isOpInGroup(op)) ||
          isa<top::InputOp, ReturnOp>(op)) {
        ops.emplace_back(op);
      }
    });
  }
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  module::getInputsOutputs(m, inputs, outputs);
  for (auto iter = ops.rbegin(); iter != ops.rend(); ++iter) {
    updateLiveRange(*iter, ops_loc, op_infos, inplace_ops, outputs,
                    neuron_alignment);
  }
  // make the order of the ops positive
  std::reverse(inplace_ops.begin(), inplace_ops.end());
  // updateConcatOpTargetV(inplace_ops, op_infos);
  for (auto iter = ops.begin(); iter != ops.end(); ++iter) {
    auto op = *iter;
    if (isa<tpu::GroupOp>(op)) {
      group_ops.emplace_back(op);
    }
    int n = op->getNumResults();
    for (int i = 0; i < n; ++i) {
      if (module::isNone(op->getResult(i))) {
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
      }
    }
  }

  int64_t sharedGmemOffset = 0;
  int64_t sharedGmemSize = 0;
  // key: the operation pointer & output index
  std::map<ValueInfo, int64_t> gaddrMap;

  for (auto &targetOuts : shared_outs_regions) {
    GmemAllocator allocator(gaddrMap, neuron_alignment);
    // FitFirstAssign should make sure op's start liverange ascendingly
    GmemAllocator::sortOpByLiveStart(targetOuts.second, liveRange);
    auto gmemUsed = allocator.assignGaddr(targetOuts.second, liveRange,
                                          reuse_addr, sharedGmemOffset);
    if (sharedGmemSize < sharedGmemOffset + gmemUsed) {
      sharedGmemSize = sharedGmemOffset + gmemUsed;
    }
  }

  int64_t baseGaddr = (((uint64_t)2) << 40);
  int64_t privateGmemSize = 0;
  // 2. Assign gaddr for ops in private region.
  if (!private_outs.empty()) {
    GmemAllocator allocator(gaddrMap, neuron_alignment);
    // FitFirstAssign should make sure op's start liverange ascendingly
    GmemAllocator::sortOpByLiveStart(private_outs, liveRange);
    privateGmemSize =
        allocator.assignGaddr(private_outs, liveRange, reuse_addr, baseGaddr);
  }

  // 3. Assign gaddr for ops in IO memory regin.
  for (int i = 0; i < (int)io_outs.size(); ++i) {
    gaddrMap[io_outs[i]] = (((uint64_t)3 + i) << 40);
  }
  // 4. set addr according to gaddrMap
  for (auto &op_addr : gaddrMap) {
    Operation *op = static_cast<Operation *>(op_addr.first.op);
    module::setAddress(op->getResult(op_addr.first.index), op_addr.second);
  }
  for (auto &v_info : inplace_ops) {
    updateAddressOfInPlaceOp(v_info, op_infos, neuron_alignment);
  }
  // 5. set in group op's addr
  for (auto &op : group_ops) {
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

  // TODO markGmemReusedOp
  // TODO crop concat pattern
  module::updateModuleTypes();
  module::setNeuronSize(m, sharedGmemSize);
  module::setGmemPrivateSize(m, privateGmemSize);
}

void CVAddressAssign::updateLiveRangeofPreOp(
    std::map<ValueInfo, OpElement> &op_infos, Operation *op, uint32_t end,
    std::map<Operation *, uint32_t> &ops_loc, MemType mem_type,
    int64_t alignment) {
  for (int i = 0; i < op->getNumOperands(); ++i) {
    auto operand = module::getOperand(op, i);
    if (module::isNone(operand)) {
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
      auto operand = module::getOperand(op, i);
      auto preOp = operand.getDefiningOp();
      ValueInfo v_info(preOp, operand.cast<OpResult>().getResultNumber());
      op_infos[v_info].live.start =
          std::min(ops_loc[preOp], op_infos[v_info].live.start);
      max_end = std::max(max_end, op_infos[v_info].live.end);
      op_infos[v_info].live.end = max_end;
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
                                      std::vector<ValueInfo> &inplace_ops,
                                      std::vector<mlir::Value> &outputs,
                                      int64_t alignment) {
  if (isa<top::InputOp>(op)) {
    ValueInfo v_info(op, 0);
    op_infos[v_info].mem_type = MEM_IOMEM;
  } else if (isa<ReturnOp>(op)) {
    MemType mem_type = MEM_PRIVATE;
    auto func_op = dyn_cast<FuncOp>(op->getParentOp());
    assert(func_op);
    if (func_op.getName() == "main") {
      mem_type = MEM_IOMEM;
    }
    updateLiveRangeofPreOp(op_infos, op, ops_loc[op] + 1, ops_loc, mem_type,
                           alignment);
  } else if (isInPlaceOp(op)) {
    ValueInfo cur_info(op, 0);
    assert(op_infos.find(cur_info) != op_infos.end());
    op_infos[cur_info].need_alloc = false;
    op_infos[cur_info].inplace = true;
    updateLiveRangeOfInPlaceOp(op_infos, op, op_infos[cur_info].live.end,
                               ops_loc, op_infos[cur_info].mem_type, alignment);
    inplace_ops.emplace_back(cur_info);
  } else if (module::isTpuOp(op)) {
    for (int i = 0; i < op->getNumResults(); ++i) {
      ValueInfo cur_info(op, i);
      if (!module::isNone(op->getResult(i))) {
        assert(op_infos.find(cur_info) != op_infos.end());
      }
    }
    updateLiveRangeofPreOp(op_infos, op, ops_loc[op] + 1, ops_loc, MEM_SHARED,
                           alignment);
  } else {
    llvm_unreachable("op not supported!");
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
    base_addr = module::getAddress(
        static_cast<Operation *>(target_v.op)->getResult(target_v.index));
    int64_t offset = 0;
    module::setAddress(op->getResult(0), base_addr + offset);
    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      auto operand = module::getOperand(op, i);
      auto opd = operand.getDefiningOp();
      if (opd == 0x0) {
        assert(0);
      }
      int this_index = operand.cast<OpResult>().getResultNumber();
      // uint32_t tensor_size = getTensorGmemSize(opd, this_index, alignment);
      uint32_t tensor_size = module::getBytes(opd->getResult(this_index));
      module::setAddress(opd->getResult(this_index), base_addr + offset);
      offset += tensor_size;
    }
  } else if (auto reshapeOp = dyn_cast<tpu::ReshapeOp>(op)) {
    auto operand = module::getOperand(op, 0);
    module::setAddress(reshapeOp.getOutput(), module::getAddress(operand));
  } else if (auto sliceOp = dyn_cast<tpu::SliceOp>(op)) {
    auto p = sliceOp.parseParam();
    int axis;
    for (axis = 0; p.offset_4[axis] == 0 && axis < 4; axis++)
      ;
    size_t offset_bytes = 0;
    if (axis != 4) {
      if (p.offset_4[axis] >= 0) {
        offset_bytes =
            p.offset_4[axis] * module::getDtypeSize(sliceOp.getOutput());
      } else {
        offset_bytes = (p.offset_4[axis] + p.is_4[axis]) *
                       module::getDtypeSize(sliceOp.getOutput());
      }
      for (int i = axis + 1; i < 4; ++i) {
        offset_bytes *= p.is_4[i];
      }
    }
    auto operand = module::getOperand(op, 0);
    module::setAddress(sliceOp.getOutput(),
                       module::getAddress(operand) + offset_bytes);
  } else {
    llvm_unreachable("set address of undefined inplace op!");
  }
}

bool CVAddressAssign::isInPlaceOp(Operation *op) {
  if (isa<tpu::ReshapeOp>(op)) {
    return true;
  } else if (auto concat_op = dyn_cast<tpu::ConcatOp>(op)) {
    if (concat_op.getOnlyMerge()) {
      return true;
    }
  } else if (auto slice_op = dyn_cast<tpu::SliceOp>(op)) {
    auto p = slice_op.parseParam();
    return p.fusible;
  } else {
    return false;
  }
  return false;
}

bool CVAddressAssign::isOutput(Operation *op, int index) {
  for (auto &use : op->getResult(index).getUses()) {
    Operation *next = use.getOwner();
    if (isa<ReturnOp>(next)) {
      return true;
    }
  }
  return false;
}

void CVAddressAssign::updateConcatOpTargetV(
    std::vector<ValueInfo> &inplace_ops,
    std::map<ValueInfo, OpElement> &op_infos) {
  for (auto iter = inplace_ops.rbegin(); iter != inplace_ops.rend(); ++iter) {
    auto value_info = *iter;
    auto op = static_cast<Operation *>(value_info.op);
    if (isa<tpu::ConcatOp>(op)) {
      auto target_v_info = op_infos[value_info].target_v;
      auto target_v_op = static_cast<Operation *>(target_v_info.op);
      if (isa<tpu::ConcatOp>(target_v_op)) {
        auto target_vv_info = op_infos[target_v_info].target_v;
        assert(op_infos.find(target_vv_info) != op_infos.end());
        if (op_infos[target_v_info].live.tensor_size >
            op_infos[target_vv_info].live.tensor_size) {
          op_infos[target_vv_info].live.tensor_size =
              op_infos[target_v_info].live.tensor_size;
        }
        if (op_infos[target_v_info].live.end >
            op_infos[target_vv_info].live.end) {
          op_infos[target_vv_info].live.end = op_infos[target_v_info].live.end;
        }
      }
    }
  }
}

uint32_t CVAddressAssign::getTensorGmemSize(Operation *op, int index,
                                            int64_t aligment_) {
  uint32_t size = module::getBytes(op->getResult(index));
  // pad to aligment_
  if (size % aligment_) {
    size = size + aligment_ - (size % aligment_);
  }
  return size;
}
} // namespace tpu
} // namespace tpu_mlir

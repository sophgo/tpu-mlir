//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LmemAllocator.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/BasicTimeStep.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <vector>

using namespace tpu_mlir::backend;

namespace tpu_mlir {
namespace tpu {

// factors used for sorting
static bool membuf_sort_std_cmp(const MemBufSortStd &membuf_a,
                                const MemBufSortStd &membuf_b) {
  bool res = false;
  if (membuf_a.first.conflict > membuf_b.first.conflict) {
    res = true;
  } else if (membuf_a.first.conflict == membuf_b.first.conflict) {
    if (membuf_a.second.area > membuf_b.second.area) {
      res = true;
    } else if (membuf_a.second.area == membuf_b.second.area) {
      res = (membuf_a.second.start_ts < membuf_b.second.start_ts);
    }
  }

  return res;
}

static inline bool is_lmem_ldst(TIMESTEP_LD_ST type) {
  return type == TIMESTEP_LOAD || type == TIMESTEP_STORE;
}

static bool is_timestep_overlapped(int64_t start_ts0, int64_t end_ts0,
                                   int64_t start_ts1, int64_t end_ts1) {
  bool flag = false;
  if (start_ts0 <= end_ts0 && start_ts1 <= end_ts1) {
    flag = !(start_ts0 > end_ts1 || start_ts1 > end_ts0);
  } else if (start_ts0 > end_ts0 && start_ts1 <= end_ts1) {
    flag = start_ts1 <= end_ts0 || start_ts0 <= end_ts1;
  } else if (start_ts0 <= end_ts0 && start_ts1 > end_ts1) {
    flag = start_ts0 <= end_ts1 || start_ts1 <= end_ts0;
  } else {
    flag = true;
  }
  return flag;
}

static int64_t get_membuf_area(int64_t start_ts, int64_t end_ts,
                               int64_t mem_size, int64_t ts_num) {
  int64_t area;
  if (start_ts <= end_ts) {
    area = (end_ts - start_ts + 1) * mem_size;
  } else {
    area = ((end_ts + 1) + (ts_num - start_ts)) * mem_size;
  }
  return area;
}

static bool is_buffer_used_by_npu(const mem_buffer_key_t &buffer_key,
                                  const TpuTsField &cur_layers) {
  if (buffer_key.type == LMEM_OPERATION) {
    return true;
  }
  auto users = buffer_key.value.getUsers();
  auto src_op = buffer_key.value.getDefiningOp();
  for (auto op : cur_layers) {
    if (src_op == op ||
        std::find(users.begin(), users.end(), op) != users.end()) {
      return true;
    }
  }
  return false;
}

static bool is_buffer_used_by_gdma(const mem_buffer_key_t &buffer_key,
                                   const GdmaTsField &cur_tensors,
                                   bool is_npu_use) {
  if (buffer_key.type != LMEM_OPERATION) {
    for (auto &tensor : cur_tensors) {
      if (tensor.first == buffer_key.value &&
          is_lmem_ldst(tensor.second.mode)) {
        if (is_npu_use && tensor.second.mode != TIMESTEP_STORE) {
          llvm::errs() << "tensor is loaded and used by npu simultaneously in "
                          "timestep\n";
          exit(-1);
        }
        return true;
      }
    }
  }
  return false;
}

static bool is_relate_op(const mem_buffer_key_t &buffer_key, Operation *op,
                         int64_t cur_ts, int64_t start_ts) {
  bool is_relate = false;
  if (buffer_key.type == LMEM_OPERATION) {
    is_relate = (op == buffer_key.op);
  } else if (cur_ts == start_ts) {
    is_relate = (op == buffer_key.value.getDefiningOp());
  } else {
    auto users = buffer_key.value.getUsers();
    is_relate = (std::find(users.begin(), users.end(), op) != users.end());
  }
  return is_relate;
}

static std::map<mem_buffer_key_t, mem_buffer_key_t *>
get_buffer_key_pointers(std::list<MemBufSortStd> &membuf_list) {
  std::map<mem_buffer_key_t, mem_buffer_key_t *> pointers;
  for (auto &iter : membuf_list) {
    pointers[iter.first] = &(iter.first);
  }
  return pointers;
}

static inline int64_t increase_nsecs(int64_t nsecs, int64_t batch_size) {
  if (nsecs == batch_size) {
    return -1;
  }
  int64_t nslice = batch_size / nsecs + (batch_size % nsecs > 0);
  int64_t new_nslice = nslice;
  int64_t next_nsecs = nsecs;
  do {
    next_nsecs++;
    new_nslice = batch_size / next_nsecs + (batch_size % next_nsecs > 0);
  } while (new_nslice >= nslice && next_nsecs < batch_size);

  return next_nsecs;
}

static inline void update_shape_secs(shape_secs_t &shape_secs,
                                     const shape_secs_t &max_shape_secs) {
  if (shape_secs.nsecs < max_shape_secs.nsecs) {
    shape_secs.nsecs = increase_nsecs(shape_secs.nsecs, max_shape_secs.nsecs);
  } else {
    ++(shape_secs.hsecs);
  }
}

void LmemAllocator::find_used_banks(std::set<int64_t> &used_banks,
                                    int64_t lmem_addr, int64_t lmem_size) {
  int64_t bank_size = Arch::LMEM_BANK_BYTES;
  int64_t start_bank = lmem_addr / bank_size;
  int64_t end_bank = (lmem_addr + lmem_size - 1) / bank_size;
  for (int64_t i = start_bank; i <= end_bank; ++i) {
    used_banks.insert(i);
  }
}

bool LmemAllocator::update_avail_lmems(std::list<MemBlock> &avail_lmems,
                                       const MemBlock &exclude_lmem) {
  bool space_split = false;
  int64_t exclude_start = exclude_lmem.first;
  int64_t exclude_end = exclude_start + exclude_lmem.second;
  std::list<MemBlock>::iterator avail_iter;
  for (avail_iter = avail_lmems.begin(); avail_iter != avail_lmems.end();) {
    int64_t avail_start = avail_iter->first;
    int64_t avail_end = avail_iter->first + avail_iter->second;
    if (avail_start >= exclude_start && avail_end <= exclude_end) {
      avail_iter = avail_lmems.erase(avail_iter);
    } else if (avail_start < exclude_start && avail_end > exclude_start &&
               avail_end <= exclude_end) {
      avail_iter->second = exclude_start - avail_start;
      avail_iter++;
    } else if (avail_start >= exclude_start && avail_start < exclude_end &&
               avail_end > exclude_end) {
      if (avail_start == exclude_start) {
        space_split = true;
      }
      avail_iter->second = avail_end - exclude_end;
      avail_iter->first = exclude_end;
      avail_iter++;
    } else if (avail_start < exclude_start && avail_end > exclude_end) {
      int new_buffer_addr = exclude_end;
      int new_buffer_size = avail_end - exclude_end;
      avail_iter->second = exclude_start - avail_start;
      avail_iter++;
      avail_lmems.insert(avail_iter,
                         std::make_pair(new_buffer_addr, new_buffer_size));
      avail_iter++;
      space_split = true;
    } else {
      avail_iter++;
    }
  }
  return space_split;
}

static bool can_membuf_inplace_alloc(int pre_start, int pre_end, int post_start, int post_end) {
  bool flag = false;
  if (post_start == pre_end) {
    if (pre_end > pre_start && post_end > post_start) {
      flag = true;
    } else if ( (pre_end < pre_start && post_end > post_start) ||
                (pre_end > pre_start && post_end < post_start) ) {
      flag = post_end < pre_start;
    }
  }
  return flag;
}

static bool isInplaceOp(Operation *op) {
  if (module::isCV18xx()) return false;
  bool flag = false;
  if (isa<tpu::ScaleOp>(op)) {
    flag = module::getStorageType(op->getOperand(0)).isF32();
  } else if (isa<tpu::AddOp, tpu::SubOp, tpu::MulOp, tpu::DivOp, tpu::MinOp, tpu::MaxOp>(op)) {
    flag = true;
    auto ins = op->getOperands();
    ValueSet ins_set(ins.begin(), ins.end());
    if (ins.size() > 2 && ins.size() != ins_set.size()) flag = false;
  } else if (isa<tpu::MulShiftOp>(op)) {
    flag = true;
    auto in = op->getOperand(0);
    if (module::isUniformQuantized(in)) {
      auto in_qtype = module::getUniformQuantizedType(in);
      auto out_qtype = module::getUniformQuantizedType(op->getResult(0));
      auto in_zp = in_qtype.getZeroPoint();
      auto out_zp = out_qtype.getZeroPoint();
      if (in_zp != 0 || out_zp != 0) flag = false;
    }
    flag = flag && !module::getStorageType(in).isF32();
  } else if (isa<tpu::ReshapeOp>(op)) {
    flag = true;
  }
  return flag && module::getBytes(op->getOperand(0)) >= module::getBytes(op->getResult(0));
}

static void insert_inplace_local_mem(const mem_buffer_key_t &buffer_key,
                                     const std::vector<mem_buffer_key_t> &ts_overlap_buffer,
                                     BasicTimeStepPtr &time_step,
                                     std::list<MemBlock> &avail_lmems) {
  // llvm::errs() << "-----------------insert_inplace_local_mem "<< buffer_key.type << "----------------------------------\n";
  if (buffer_key.type == LMEM_OPERATION)
    return;
  auto &buffer_value = time_step->get_lmem_buffer_value(buffer_key);
  auto value = buffer_key.value;
  int64_t buffer_idx = 0;
  MemBlock lmem_locate(0, 0);
  bool inplace_valid;
  auto src_layer = value.getDefiningOp();
  if (src_layer) {
    if (isa<top::WeightOp>(src_layer)) {
      // llvm::errs() << "-----------------Oh no it is a weight----------------------------------\n";
      return;
    }
    if (isInplaceOp(src_layer)) {
      // llvm::errs() << "-----------------src-" << src_layer->getName() <<"----------------------------------\n";
      auto src_in = src_layer->getOperand(0);
      for(buffer_idx = 0; buffer_idx < ts_overlap_buffer.size(); ++buffer_idx) {
        auto &src_buffer_value = time_step->get_lmem_buffer_value(ts_overlap_buffer[buffer_idx]);
        if (ts_overlap_buffer[buffer_idx].type != LMEM_OPERATION &&
            ts_overlap_buffer[buffer_idx].value == src_in &&
            src_buffer_value.end_ts == buffer_value.start_ts) {
          // llvm::errs() << "-----------------if-----------------------------------\n";
          if (can_membuf_inplace_alloc(src_buffer_value.start_ts, src_buffer_value.end_ts, buffer_value.start_ts, buffer_value.end_ts)) {
            lmem_locate.first = src_buffer_value.addr;
            lmem_locate.second = src_buffer_value.size;
            // llvm::errs() << "-----------------can alloc-----------------------------------\n";
          } else {
            // llvm::errs() << "-----------------no can alloc-----------------------------------\n";
            buffer_idx = ts_overlap_buffer.size();
          }
          break;
        }
      }

      if (buffer_idx != ts_overlap_buffer.size()) {
        inplace_valid = true;
        for(int64_t i = 0; i < ts_overlap_buffer.size(); ++i) {
          if (buffer_idx == i) continue;
          auto &buffer_value0 = time_step->get_lmem_buffer_value(ts_overlap_buffer[i]);
          if ( !(buffer_value0.addr >= (lmem_locate.first + lmem_locate.second) ||
                (buffer_value0.addr + buffer_value0.size) <= lmem_locate.first) ) {
            inplace_valid = false;
            break;
            // llvm::errs() << "-----------------invalid-----------------------------------\n";
          }
        }
        if (inplace_valid) {
          auto list_it = avail_lmems.begin();
          for(; list_it != avail_lmems.end(); ++list_it) {
            if (list_it->first > lmem_locate.first) {
              break;
            }
          }
          // llvm::errs() << "+++++++++++++++++++++++++" << lmem_locate.first << ", " << lmem_locate.second <<"----------------------------------\n";
          avail_lmems.insert(list_it, lmem_locate);
        }
      }
    }
  }

  auto dst_layers = value.getUsers();
  if (!dst_layers.empty()) {
    auto dst_layer = *dst_layers.begin();
    if (isInplaceOp(dst_layer) && value == dst_layer->getOperand(0)) {
      // llvm::errs() << "-----------------dst-" << dst_layer->getName() <<"----------------------------------\n";
      auto dst_out = dst_layer->getResult(0);
      for (buffer_idx = 0; buffer_idx < ts_overlap_buffer.size(); ++buffer_idx) {
        auto &dst_buffer_value = time_step->get_lmem_buffer_value(ts_overlap_buffer[buffer_idx]);
        if (ts_overlap_buffer[buffer_idx].type != LMEM_OPERATION &&
            ts_overlap_buffer[buffer_idx].value == dst_out &&
            dst_buffer_value.start_ts == buffer_value.end_ts) {
          // llvm::errs() << "-----------------if-----------------------------------\n";
          if (can_membuf_inplace_alloc(buffer_value.start_ts, buffer_value.end_ts, dst_buffer_value.start_ts, dst_buffer_value.end_ts)) {
            lmem_locate.first = dst_buffer_value.addr;
            lmem_locate.second = dst_buffer_value.size;
            // llvm::errs() << "-----------------can alloc-----------------------------------\n";
          } else {
            // llvm::errs() << "-----------------no can alloc-----------------------------------\n";
            buffer_idx = ts_overlap_buffer.size();
          }
          break;
        }
      }

      if (buffer_idx != ts_overlap_buffer.size()) {
        inplace_valid = true;
        for(int64_t i = 0; i < ts_overlap_buffer.size(); ++i) {
          if (buffer_idx == i) continue;
          auto &buffer_value1 = time_step->get_lmem_buffer_value(ts_overlap_buffer[i]);
          if ( !(buffer_value1.addr >= (lmem_locate.first + lmem_locate.second) ||
                (buffer_value1.addr + buffer_value1.size) <= lmem_locate.first) ) {
            inplace_valid = false;
            // llvm::errs() << "-----------------invalid-----------------------------------\n";
            break;
          }
        }
        if (inplace_valid) {
          auto list_it = avail_lmems.begin();
          for(; list_it != avail_lmems.end(); ++list_it) {
            if (list_it->first > lmem_locate.first) {
              break;
            }
          }
          // llvm::errs() << "++++++++++++++++++++++++++" << lmem_locate.first << ", " << lmem_locate.second <<"----------------------------------\n";
          avail_lmems.insert(list_it, lmem_locate);
        }
      }
    }
  }
}

void LmemAllocator::update_avail_lmems(
    std::list<MemBlock> &avail_lmems, const mem_buffer_key_t &buffer_key,
    const mem_buffer_value_t &buffer_value,
    const mem_buffer_key_t &recent_buffer_allocated,
    const mem_buffer_value_t &recent_buffer_value, BasicTimeStepPtr &time_step,
    bool hold_on_coeff, bool consider_inplace) {
  // find the allocated buffer overlap in time dimension
  bool ts_overlap = is_timestep_overlapped(
      buffer_value.start_ts, buffer_value.end_ts, recent_buffer_value.start_ts,
      recent_buffer_value.end_ts);

  if (!ts_overlap && hold_on_coeff) {
    ts_overlap =
        (buffer_key.type != LMEM_OPERATION &&
         time_step->is_tensor_hold_in_lmem(buffer_key.value)) ||
        (recent_buffer_allocated.type != LMEM_OPERATION &&
         time_step->is_tensor_hold_in_lmem(recent_buffer_allocated.value));
  }

  if (ts_overlap) {
    MemBlock used_overlap_buffer;
    used_overlap_buffer.first = recent_buffer_value.addr;
    used_overlap_buffer.second = recent_buffer_value.size;
    used_overlap_buffer.second =
        align_up(used_overlap_buffer.first + used_overlap_buffer.second,
                 Arch::EU_BYTES) -
        used_overlap_buffer.first;
    // buffer_key.align_bytes

    bool space_split = update_avail_lmems(avail_lmems, used_overlap_buffer);

    if (consider_inplace && space_split) {
      // EXPERIMENTAL FEATURE
      // may cause performance up or down simultaneously
      // comment out the following lines if encounter performance issues
      std::vector<mem_buffer_key_t> ts_overlap_buffer(1,
                                                      recent_buffer_allocated);
      insert_inplace_local_mem(buffer_key, ts_overlap_buffer,
                               time_step, avail_lmems);
    }
  }

  // delete the available memory buffer that is smaller than requirement
  std::list<MemBlock>::iterator avail_iter;
  for (avail_iter = avail_lmems.begin(); avail_iter != avail_lmems.end();) {
    if (avail_iter->second < buffer_value.size) {
      avail_iter = avail_lmems.erase(avail_iter);
    } else {
      avail_iter++;
    }
  }
}

MemBlock LmemAllocator::find_avail_lmem_location(
    avail_space_t &avail_space, const mem_buffer_key_t &buffer_key,
    const mem_buffer_value_t &buffer_value) {

  // refine avail_lmems according to exclude_banks
  MemBlock bank_lmem;
  std::list<MemBlock> avail_lmems_tmp = avail_space.avail_lmems;
  for (auto bank_idx : avail_space.exclude_banks) {
    bank_lmem.first = bank_idx * Arch::LMEM_BANK_BYTES;
    bank_lmem.second = Arch::LMEM_BANK_BYTES;
    update_avail_lmems(avail_lmems_tmp, bank_lmem);
  }

  MemBlock alloc_lmem(-1, -1);
  for (auto avail_iter = avail_lmems_tmp.begin();
       avail_iter != avail_lmems_tmp.end(); ++avail_iter) {
    if (avail_iter->second >= buffer_value.size) {
      alloc_lmem = *avail_iter;
      break;
    }
  }

  // allow bank confict if could not find space not conflict
  if (alloc_lmem.first == -1 && !avail_space.avail_lmems.empty()) {
    alloc_lmem = avail_space.avail_lmems.front();
  }

  return alloc_lmem;
}

void LmemAllocator::update_exclude_banks(
    std::set<int64_t> &exclude_banks, const mem_buffer_key_t &buffer_key,
    const mem_buffer_value_t &buffer_value,
    const mem_buffer_key_t &recent_buffer_allocated,
    const mem_buffer_value_t &recent_buffer_value,
    BasicTimeStepPtr &time_step) {
  int64_t timestep_num = time_step->get_timestep_num();

  bool first_step = true;
  bool is_npu_use, is_gdma_use;
  bool is_recent_used_banks_updated = false;
  std::set<int64_t> recent_used_banks;
  for (int64_t ts = buffer_value.start_ts;
       (ts != ((buffer_value.end_ts + 1) % timestep_num)) || first_step;
       ts = (ts + 1) % timestep_num) {
    first_step = false;
    const TpuTsField &cur_layers = time_step->getLayers(ts);
    const GdmaTsField &cur_tensors = time_step->getTensors(ts);
    is_npu_use = is_buffer_used_by_npu(buffer_key, cur_layers);
    is_gdma_use = is_buffer_used_by_gdma(buffer_key, cur_tensors, is_npu_use);

    // find the banks that have been used by npu if the current buffer is used
    // by gdma
    if (is_gdma_use || is_npu_use) {
      for (auto op : cur_layers) {
        if (is_npu_use &&
            !is_relate_op(buffer_key, op, ts, buffer_value.start_ts)) {
          continue;
        }
        if (is_relate_op(recent_buffer_allocated, op, ts,
                         recent_buffer_value.start_ts)) {
          find_used_banks(recent_used_banks, recent_buffer_value.addr,
                          recent_buffer_value.size);
          is_recent_used_banks_updated = true;
          break;
        }
      }
    }

    // find the banks that have been used by gmda if the current buffer is
    // used by npu
    if (!is_recent_used_banks_updated && is_npu_use &&
        is_buffer_used_by_gdma(recent_buffer_allocated, cur_tensors, false)) {
      find_used_banks(recent_used_banks, recent_buffer_value.addr,
                      recent_buffer_value.size);
      is_recent_used_banks_updated = true;
    }

    if (is_recent_used_banks_updated) {
      break;
    }
  }

  exclude_banks.insert(recent_used_banks.begin(), recent_used_banks.end());
}

MemBlock LmemAllocator::global_find_avail_lmem_localtion(
    avail_space_t &avail_space, const mem_buffer_key_t &buffer_key,
    const mem_buffer_key_t &recent_buffer_allocated,
    BasicTimeStepPtr &time_step, bool one_loop) {
  auto &buffer_value = time_step->get_lmem_buffer_value(buffer_key);
  auto &recent_buffer_value =
      time_step->get_lmem_buffer_value(recent_buffer_allocated);
  update_exclude_banks(avail_space.exclude_banks, buffer_key, buffer_value,
                       recent_buffer_allocated, recent_buffer_value, time_step);

  update_avail_lmems(avail_space.avail_lmems, buffer_key, buffer_value,
                     recent_buffer_allocated, recent_buffer_value, time_step,
                     !one_loop, true);

  // get the available local memory location
  auto alloc_mem =
      find_avail_lmem_location(avail_space, buffer_key, buffer_value);

  return alloc_mem;
}

void membuf_heap_create(
    std::vector<std::set<mem_buffer_key_t *>> &npu_membuf_heap,
    std::vector<std::set<mem_buffer_key_t *>> &gdma_membuf_heap,
    std::list<MemBufSortStd> &membuf_list, BasicTimeStepPtr &time_step) {
  // auto &mem_buff = time_step->get_lmem_buffer();
  int64_t timestep_num = time_step->get_timestep_num();
  auto pointers = get_buffer_key_pointers(membuf_list);
  std::set<mem_buffer_key_t *> membuf_heap;
  mem_buffer_key_t elt;

  for (int64_t ts = 0; ts < timestep_num; ++ts) {
    auto &ts_layers = time_step->getLayers(ts);
    auto &ts_tensors = time_step->getTensors(ts);

    if (!ts_layers.empty()) {
      // bdc
      membuf_heap.clear();
      for (size_t i = 0; i < ts_layers.size(); ++i) {
        auto op = ts_layers[i];
        for (auto in : op->getOperands()) {
          if (in.getType().isa<NoneType>()) {
            continue;
          }
          elt.value = in;
          elt.type = (in.getDefiningOp() != nullptr &&
                      isa<top::WeightOp>(in.getDefiningOp()))
                         ? LMEM_WEIGHT
                         : LMEM_ACTIVATION;
          membuf_heap.insert(pointers[elt]);
        }

        for (auto out : get_output_values(op)) {
          elt.value = out;
          elt.type = LMEM_ACTIVATION;
          membuf_heap.insert(pointers[elt]);
        }

        // A performance drop was found in yolov5s when considering
        // imm_buffer. More cases need to be tested.
        // // imm_buffer
        // elt.op = op;
        // elt.type = LMEM_OPERATION;
        // if (mem_buff.find(elt) != mem_buff.end()) {
        //   membuf_heap.insert(pointers[elt]);
        // }
      }
      npu_membuf_heap.push_back(membuf_heap);

      // gdma
      membuf_heap.clear();
      for (size_t i = 0; i < ts_tensors.size(); ++i) {
        if (is_lmem_ldst(ts_tensors[i].second.mode)) {
          elt.value = ts_tensors[i].first;
          if (elt.value.getDefiningOp() &&
              isa<top::WeightOp>(elt.value.getDefiningOp())) {
            elt.type = LMEM_WEIGHT;
          } else {
            elt.type = LMEM_ACTIVATION;
          }
          membuf_heap.insert(pointers[elt]);
        }
      }
      gdma_membuf_heap.push_back(membuf_heap);
    } else {
      membuf_heap.clear();
      npu_membuf_heap.push_back(membuf_heap);
      gdma_membuf_heap.push_back(membuf_heap);
    }
  }
}

void update_membuf_conflict_param(
    std::vector<std::set<mem_buffer_key_t *>> &npu_membuf_heap,
    std::vector<std::set<mem_buffer_key_t *>> &gdma_membuf_heap,
    std::list<MemBufSortStd> &membuf_list) {
  // reset all conflict
  for (auto &iter : membuf_list) {
    iter.first.conflict = 0;
  }
  // update all conflict
  for (size_t i = 0; i < npu_membuf_heap.size(); ++i) {
    for (auto p_key : npu_membuf_heap[i]) {
      p_key->conflict = 1;
    }
  }
  for (size_t i = 0; i < gdma_membuf_heap.size(); ++i) {
    for (auto p_key : gdma_membuf_heap[i]) {
      p_key->conflict = 1;
    }
  }
}

static void conflict_heap_delete(
    std::vector<std::set<mem_buffer_key_t *>> &npu_membuf_heap,
    std::vector<std::set<mem_buffer_key_t *>> &gdma_membuf_heap,
    mem_buffer_key_t *delete_element) {
  std::set<mem_buffer_key_t *>::iterator iter;
  for (size_t i = 0; i < npu_membuf_heap.size(); ++i) {
    iter = npu_membuf_heap[i].find(delete_element);
    if (iter != npu_membuf_heap[i].end()) {
      npu_membuf_heap[i].erase(iter);
    }
    if (npu_membuf_heap[i].empty() && !gdma_membuf_heap[i].empty()) {
      gdma_membuf_heap[i].clear();
    }
  }
  for (size_t i = 0; i < gdma_membuf_heap.size(); ++i) {
    iter = gdma_membuf_heap[i].find(delete_element);
    if (iter != gdma_membuf_heap[i].end()) {
      gdma_membuf_heap[i].erase(iter);
    }
    if (npu_membuf_heap[i].size() == 1 && gdma_membuf_heap[i].empty()) {
      npu_membuf_heap[i].clear();
    }
  }
}

static void init_membuf_list(std::list<MemBufSortStd> &membuf_list,
                             const BasicTimeStepPtr &time_step, bool one_loop) {
  int64_t membuf_area;
  bool hold_on_coeff;
  membuf_sort_std_t membuf_sort_std;
  int64_t timestep_num = time_step->get_timestep_num();
  const MemBuff &mem_buffer = time_step->get_lmem_buffer();
  for (auto it = mem_buffer.begin(); it != mem_buffer.end(); ++it) {
    hold_on_coeff = false;
    if (it->first.type != LMEM_OPERATION) {
      Value v = it->first.value;
      hold_on_coeff = time_step->is_tensor_hold_in_lmem(v);
    }

    if (!one_loop && hold_on_coeff) {
      membuf_area = timestep_num * it->second.size;
    } else {
      membuf_area = get_membuf_area(it->second.start_ts, it->second.end_ts,
                                    it->second.size, timestep_num);
    }
    // membuf_sort_std.conflict = 0;
    membuf_sort_std.area = membuf_area;
    membuf_sort_std.start_ts = it->second.start_ts;
    membuf_list.push_back(std::make_pair(it->first, membuf_sort_std));
  }
}

void init_buffer_avail_space(BufferAvailSpace &buffer_avail_space,
                             std::list<MemBufSortStd> &membuf_list) {
  avail_space_t avail_space;
  for (auto buflist_it = membuf_list.begin(); buflist_it != membuf_list.end();
       ++buflist_it) {
    avail_space.avail_lmems.clear();
    avail_space.avail_lmems.push_back(std::make_pair(0, Arch::LMEM_BYTES));
    buffer_avail_space.insert(std::make_pair(buflist_it->first, avail_space));
  }
}

bool LmemAllocator::assignLmemAddr(const LgInfo &lg_info,
                                   BasicTimeStepPtr &time_step,
                                   const shape_secs_t &shape_secs) {
  time_step->update_all_mem_buffer_size(lg_info);
  bool one_loop = (shape_secs.nsecs == 1 && shape_secs.hsecs == 1);

  std::list<MemBufSortStd> membuf_list;
  init_membuf_list(membuf_list, time_step, one_loop);

  // init avail_lmems and exclude_banks
  BufferAvailSpace buffer_avail_space;
  init_buffer_avail_space(buffer_avail_space, membuf_list);

  // create conflict heap
  std::vector<std::set<mem_buffer_key_t *>> npu_membuf_heap;
  std::vector<std::set<mem_buffer_key_t *>> gdma_membuf_heap;
  membuf_heap_create(npu_membuf_heap, gdma_membuf_heap, membuf_list, time_step);

  MemBlock alloc_lmem; // consider use alloc_position instead
  int64_t tgt_position = 0;
  int64_t lmem_occupy = 0;
  bool first_alloc = true;
  mem_buffer_key_t recent_buffer_allocated;
  std::list<MemBufSortStd>::iterator buflist_it;
  std::list<MemBufSortStd>::iterator tgt_buflist_it;
  while (!membuf_list.empty()) {
    tgt_position = Arch::LMEM_BYTES;
    update_membuf_conflict_param(npu_membuf_heap, gdma_membuf_heap,
                                 membuf_list);
    membuf_list.sort(membuf_sort_std_cmp);

    for (buflist_it = membuf_list.begin(); buflist_it != membuf_list.end();
         ++buflist_it) {
      if (first_alloc) {
        first_alloc = false;
        if (time_step->get_lmem_size(buflist_it->first) <= Arch::LMEM_BYTES) {
          tgt_position = 0;
          tgt_buflist_it = buflist_it;
        } else {
          return false;
        }
        break;
      } else {
        alloc_lmem = global_find_avail_lmem_localtion(
            buffer_avail_space[buflist_it->first], buflist_it->first,
            recent_buffer_allocated, time_step, one_loop);
        if (alloc_lmem.first != -1) {
          if (alloc_lmem.first < tgt_position) {
            tgt_position = alloc_lmem.first;
            tgt_buflist_it = buflist_it;
          }
        } else {
          return false;
        }
      }
    }

    if (tgt_position < Arch::LMEM_BYTES) {
      recent_buffer_allocated = tgt_buflist_it->first;
      time_step->set_lmem_addr(tgt_buflist_it->first, tgt_position);
      int64_t buffer_end =
          tgt_position + time_step->get_lmem_size(tgt_buflist_it->first);
      lmem_occupy = buffer_end > lmem_occupy ? buffer_end : lmem_occupy;
      conflict_heap_delete(npu_membuf_heap, gdma_membuf_heap,
                           &(tgt_buflist_it->first));
      membuf_list.erase(tgt_buflist_it);
      buffer_avail_space.erase(tgt_buflist_it->first);
    } else {
      llvm::errs() << "Cannot find local memory location for memory buffers\n";
      return false;
    }
  }

  time_step->set_lmem_occupy(lmem_occupy);
  return true;
}

bool LmemAllocator::assignLmemAddrWithSecs(const LgInfo &lg_info,
                                           BasicTimeStepPtr &time_step,
                                           shape_secs_t &shape_secs) {
  shape_secs_t max_shape_secs = get_group_max_secs(lg_info);
  update_data_split(time_step, lg_info, shape_secs);
//  if (assignLmemAddr(lg_info, time_step, shape_secs)) {
//    return true;
//  }
//  update_shape_secs(shape_secs, max_shape_secs);

  int64_t try_num = 0;
  bool status = false;
  const int64_t MAX_TRY_NUM = 20;
  while (shape_secs.nsecs <= max_shape_secs.nsecs &&
         shape_secs.hsecs <= max_shape_secs.hsecs) {
    // reassign time step
    status = time_step->assignTimeStep(lg_info, shape_secs, true);
    if (status == false) {
      return false;
    }
    status = assignLmemAddr(lg_info, time_step, shape_secs);

    if (status == false) {
      update_shape_secs(shape_secs, max_shape_secs);
    } else {
      break;
    }
    if (++try_num >= MAX_TRY_NUM) {
      return false;
    }
  }
  return status;
}

/// The pass for local memory allocation
class LocalMemoryAllocationPass : public LgPass {
public:
  virtual bool run(LgPassIR *pass_ir) override {
    for (size_t i = 0; i < pass_ir->lg_infos.size(); ++i) {
      if (pass_ir->lg_infos[i].group_ops.size() > 1) {
        auto lmem_allocator = LmemAllocator();
        auto ret = lmem_allocator.assignLmemAddrWithSecs(
            pass_ir->lg_infos[i], pass_ir->time_steps[i],
            pass_ir->shape_secs[i]);
        if (!ret) {
          llvm::errs() << "local memory allocate failed for group " << i
                       << "\n";
          return false;
        }
      }
    }
    return true;
  }
  virtual std::string name() override { return "LocalMemoryAllocationPass"; }
  virtual std::string brief() override {
    return "Allocate local memory for all layer groups";
  }
};

std::unique_ptr<LgPass> CreateLocalMemoryAllocationPass() {
  return std::unique_ptr<LgPass>(new LocalMemoryAllocationPass());
}

} // namespace tpu
} // namespace tpu_mlir

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LmemAllocator.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/CycleCalculator.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include "tpu_mlir/Support/Logger.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <llvm/Support/Debug.h>
#include <unordered_map>
#define DEBUG_TYPE "layer-group"

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
  DEBUG_WITH_TYPE("better_slice", {
    // log to monitor when to find next_nslices with better slice
    if (new_nslice < nslice) {
      llvm::dbgs() << "; action = better_slice"
                   << "; step = increase_nsecs"
                   << "; new_nslice = " << new_nslice << "; nslice = " << nslice
                   << "\n";
    }
  });
  return next_nsecs;
}

static inline int64_t increase_csecs(int64_t csecs, int64_t max_csecs) {
  if (csecs == max_csecs) {
    return -1;
  }
  int64_t cslice = max_csecs / csecs + (max_csecs % csecs > 0);
  int64_t new_cslice = cslice;
  int64_t next_csecs = csecs;
  do {
    next_csecs++;
    new_cslice = max_csecs / next_csecs + (max_csecs % next_csecs > 0);
  } while (new_cslice >= cslice && next_csecs < max_csecs);

  DEBUG_WITH_TYPE("better_slice", {
    // log to monitor when to find next_csecs with better slice
    if (new_cslice < cslice) {
      llvm::dbgs() << "; action = better_slice"
                   << "; step = increase_csecs"
                   << "; new_cslice = " << new_cslice << "; cslice = " << cslice
                   << "\n";
    }
  });
  return next_csecs;
}

static inline void update_shape_secs(const LgInfo &lg_info,
                                     shape_secs_t &shape_secs,
                                     int64_t &dhw_secs,
                                     const shape_secs_t &max_shape_secs) {
  if (shape_secs.nsecs < max_shape_secs.nsecs) {
    shape_secs.nsecs = increase_nsecs(shape_secs.nsecs, max_shape_secs.nsecs);
  } else if (shape_secs.csecs < max_shape_secs.csecs) {
    shape_secs.csecs = increase_csecs(shape_secs.csecs, max_shape_secs.csecs);
  } else {
    assign_dhwsecs(lg_info, shape_secs, ++dhw_secs, max_shape_secs);
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

static bool can_membuf_inplace_alloc(int pre_start, int pre_end, int post_start,
                                     int post_end) {
  bool flag = false;
  if (post_start == pre_end) {
    if (pre_end > pre_start && post_end > post_start) {
      flag = true;
    } else if ((pre_end < pre_start && post_end > post_start) ||
               (pre_end > pre_start && post_end < post_start)) {
      flag = post_end < pre_start;
    }
  }
  return flag;
}

static bool isInplaceOp(Operation *op) {
  if (module::isCV18xx())
    return false;
  if(op->getNumOperands() > 0) {
    auto parent = op->getOperand(0).getUsers();
    int users_num = std::distance(parent.begin(), parent.end());
    if(users_num > 1){
      return false;
    }
  }
  bool flag = false;
  if (isa<tpu::ScaleOp>(op)) {
    flag = module::getStorageType(op->getOperand(0)).isF32();
  } else if (isa<tpu::AddOp, tpu::SubOp, tpu::MulOp, tpu::DivOp, tpu::MinOp,
                 tpu::MaxOp>(op)) {
    flag = true;
    auto ins = op->getOperands();
    ValueSet ins_set(ins.begin(), ins.end());
    if (ins.size() > 2 && ins.size() != ins_set.size())
      flag = false;
  } else if (isa<tpu::MulShiftOp>(op)) {
    flag = true;
    auto in = op->getOperand(0);
    if (module::isUniformQuantized(in)) {
      auto in_qtype = module::getUniformQuantizedType(in);
      auto out_qtype = module::getUniformQuantizedType(op->getResult(0));
      auto in_zp = in_qtype.getZeroPoint();
      auto out_zp = out_qtype.getZeroPoint();
      if (in_zp != 0 || out_zp != 0)
        flag = false;
    }
    flag = flag && !module::getStorageType(in).isF32();
  } else if (isa<tpu::ReshapeOp>(op)) {
    flag = true;
  }
  return flag && module::getBytes(op->getOperand(0)) >=
                     module::getBytes(op->getResult(0));
}

static void
insert_inplace_local_mem(const mem_buffer_key_t &buffer_key,
                         const std::vector<mem_buffer_key_t> &ts_overlap_buffer,
                         BasicTimeStepPtr &time_step,
                         std::list<MemBlock> &avail_lmems) {
  // llvm::errs() << "-----------------insert_inplace_local_mem "<<
  // buffer_key.type << "----------------------------------\n";
  if (buffer_key.type == LMEM_OPERATION)
    return;
  auto &buffer_value = time_step->get_lmem_buffer_value(buffer_key);
  auto value = buffer_key.value;
  int64_t buffer_idx = 0;
  MemBlock lmem_locate(0, 0);
  bool inplace_valid;
  bool need_store = false;
  auto src_layer = value.getDefiningOp();
  if (src_layer) {
    if (isa<top::WeightOp>(src_layer)) {
      // llvm::errs() << "-----------------Oh no it is a
      // weight----------------------------------\n";
      return;
    }
    if (isInplaceOp(src_layer)) {
      // llvm::errs() << "-----------------src-" << src_layer->getName()
      // <<"----------------------------------\n";
      auto src_in = src_layer->getOperand(0);
      for (buffer_idx = 0; buffer_idx < ts_overlap_buffer.size();
           ++buffer_idx) {
        auto &src_buffer_value =
            time_step->get_lmem_buffer_value(ts_overlap_buffer[buffer_idx]);
        auto &tensors = time_step->getTensors(src_buffer_value.end_ts);
        for (auto t : tensors) {
          if (ts_overlap_buffer[buffer_idx].value == t.first &&
              t.second.mode == TIMESTEP_STORE) {
            need_store = true;
          }
        }
        if (ts_overlap_buffer[buffer_idx].type != LMEM_OPERATION &&
            ts_overlap_buffer[buffer_idx].value == src_in &&
            src_buffer_value.end_ts == buffer_value.start_ts &&
            !need_store) {
          // llvm::errs() <<
          // "-----------------if-----------------------------------\n";
          if (can_membuf_inplace_alloc(
                  src_buffer_value.start_ts, src_buffer_value.end_ts,
                  buffer_value.start_ts, buffer_value.end_ts)) {
            lmem_locate.first = src_buffer_value.addr;
            lmem_locate.second = src_buffer_value.size;
            // llvm::errs() << "-----------------can
            // alloc-----------------------------------\n";
          } else {
            // llvm::errs() << "-----------------no can
            // alloc-----------------------------------\n";
            buffer_idx = ts_overlap_buffer.size();
          }
          break;
        }
      }

      if (buffer_idx != ts_overlap_buffer.size()) {
        inplace_valid = true;
        for (int64_t i = 0; i < ts_overlap_buffer.size(); ++i) {
          if (buffer_idx == i)
            continue;
          auto &buffer_value0 =
              time_step->get_lmem_buffer_value(ts_overlap_buffer[i]);
          if (!(buffer_value0.addr >=
                    (lmem_locate.first + lmem_locate.second) ||
                (buffer_value0.addr + buffer_value0.size) <=
                    lmem_locate.first)) {
            inplace_valid = false;
            break;
            // llvm::errs() <<
            // "-----------------invalid-----------------------------------\n";
          }
        }
        if (inplace_valid) {
          auto list_it = avail_lmems.begin();
          for (; list_it != avail_lmems.end(); ++list_it) {
            if (list_it->first > lmem_locate.first) {
              break;
            }
          }
          // llvm::errs() << "+++++++++++++++++++++++++" << lmem_locate.first <<
          // ", " << lmem_locate.second
          // <<"----------------------------------\n";
          avail_lmems.insert(list_it, lmem_locate);
        }
      }
    }
  }

  auto dst_layers = value.getUsers();
  if (!dst_layers.empty()) {
    auto dst_layer = *dst_layers.begin();
    if (isInplaceOp(dst_layer) && value == dst_layer->getOperand(0)) {
      // llvm::errs() << "-----------------dst-" << dst_layer->getName()
      // <<"----------------------------------\n";
      auto dst_out = dst_layer->getResult(0);
      for (buffer_idx = 0; buffer_idx < ts_overlap_buffer.size();
           ++buffer_idx) {
        auto &dst_buffer_value =
            time_step->get_lmem_buffer_value(ts_overlap_buffer[buffer_idx]);
        if (ts_overlap_buffer[buffer_idx].type != LMEM_OPERATION &&
            ts_overlap_buffer[buffer_idx].value == dst_out &&
            dst_buffer_value.start_ts == buffer_value.end_ts) {
          // llvm::errs() <<
          // "-----------------if-----------------------------------\n";
          if (can_membuf_inplace_alloc(
                  buffer_value.start_ts, buffer_value.end_ts,
                  dst_buffer_value.start_ts, dst_buffer_value.end_ts)) {
            lmem_locate.first = dst_buffer_value.addr;
            lmem_locate.second = dst_buffer_value.size;
            // llvm::errs() << "-----------------can
            // alloc-----------------------------------\n";
          } else {
            // llvm::errs() << "-----------------no can
            // alloc-----------------------------------\n";
            buffer_idx = ts_overlap_buffer.size();
          }
          break;
        }
      }

      if (buffer_idx != ts_overlap_buffer.size()) {
        inplace_valid = true;
        for (int64_t i = 0; i < ts_overlap_buffer.size(); ++i) {
          if (buffer_idx == i)
            continue;
          auto &buffer_value1 =
              time_step->get_lmem_buffer_value(ts_overlap_buffer[i]);
          if (!(buffer_value1.addr >=
                    (lmem_locate.first + lmem_locate.second) ||
                (buffer_value1.addr + buffer_value1.size) <=
                    lmem_locate.first)) {
            inplace_valid = false;
            // llvm::errs() <<
            // "-----------------invalid-----------------------------------\n";
            break;
          }
        }
        if (inplace_valid) {
          auto list_it = avail_lmems.begin();
          for (; list_it != avail_lmems.end(); ++list_it) {
            if (list_it->first > lmem_locate.first) {
              break;
            }
          }
          // llvm::errs() << "++++++++++++++++++++++++++" << lmem_locate.first
          // << ", " << lmem_locate.second
          // <<"----------------------------------\n";
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
  PROFILE_LOG("update_avail_lmems", true);
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
      insert_inplace_local_mem(buffer_key, ts_overlap_buffer, time_step,
                               avail_lmems);
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
  PROFILE_LOG("update_avail_lmems", false);
}

MemBlock LmemAllocator::find_avail_lmem_location(
    avail_space_t &avail_space, const mem_buffer_key_t &buffer_key,
    const mem_buffer_value_t &buffer_value,
    bool allow_bank_conflict) {

  MemBlock alloc_lmem(-1, -1);
  if (avail_space.avail_lmems.empty()) {
    DEBUG_WITH_TYPE("assign_lmem", {
      llvm::dbgs() << "; action = find_avail_lmem"
                   << "; step = avail_lmems_empty" << "\n";
    });
    return alloc_lmem;
  }

  if (allow_bank_conflict) {
    alloc_lmem = avail_space.avail_lmems.front();
    DEBUG_WITH_TYPE("assign_lmem", {
      llvm::dbgs() << "; action = find_avail_lmem"
                   << "; step = use_bank_conflict_buffer"
                   << "; lmem = " << alloc_lmem.first
                   << "; size = " << alloc_lmem.second << "\n";
    });
    return alloc_lmem;
  }

  // refine avail_lmems according to exclude_banks
  MemBlock bank_lmem;
  std::list<MemBlock> avail_lmems_tmp = avail_space.avail_lmems;
  for (auto bank_idx : avail_space.exclude_banks) {
    bank_lmem.first = bank_idx * Arch::LMEM_BANK_BYTES;
    bank_lmem.second = Arch::LMEM_BANK_BYTES;
    update_avail_lmems(avail_lmems_tmp, bank_lmem);
  }

  for (auto avail_iter = avail_lmems_tmp.begin();
       avail_iter != avail_lmems_tmp.end(); ++avail_iter) {
    DEBUG_WITH_TYPE("assign_lmem", {
      llvm::dbgs() << "; action = find_avail_lmem" << "; step = iter_avail_lmem"
                   << "; lmem = " << avail_iter->first
                   << "; size = " << avail_iter->second << "\n";
    });
    if (avail_iter->second >= buffer_value.size) {
      alloc_lmem = *avail_iter;
      DEBUG_WITH_TYPE("assign_lmem", {
        llvm::dbgs() << "; action = find_avail_lmem"
                     << "; step = find_availble_buffer"
                     << "; lmem = " << alloc_lmem.first
                     << "; size = " << alloc_lmem.second << "\n";
      });
      break;
    }
  }

  // allow bank confict if could not find space not conflict
  if (alloc_lmem.first == -1) {
    alloc_lmem = avail_space.avail_lmems.front();
    DEBUG_WITH_TYPE("assign_lmem", {
      llvm::dbgs() << "; action = find_avail_lmem"
                   << "; step = use_bank_conflict_buffer"
                   << "; lmem = " << alloc_lmem.first
                   << "; size = " << alloc_lmem.second << "\n";
    });
  }

  return alloc_lmem;
}

void LmemAllocator::update_exclude_banks(
    std::set<int64_t> &exclude_banks, const mem_buffer_key_t &buffer_key,
    const mem_buffer_value_t &buffer_value,
    const mem_buffer_key_t &recent_buffer_allocated,
    const mem_buffer_value_t &recent_buffer_value,
    BasicTimeStepPtr &time_step) {
  PROFILE_LOG("update_exclude_banks", true);
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
  PROFILE_LOG("update_exclude_banks", false);
}

MemBlock LmemAllocator::global_find_avail_lmem_localtion(
    avail_space_t &avail_space, const mem_buffer_key_t &buffer_key,
    const mem_buffer_key_t &recent_buffer_allocated,
    BasicTimeStepPtr &time_step, bool one_loop,
    bool allow_bank_conflict) {
  PROFILE_LOG("global_find_avail_lmem_localtion", true);
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
      find_avail_lmem_location(avail_space, buffer_key, buffer_value,
                               allow_bank_conflict);
  PROFILE_LOG("global_find_avail_lmem_localtion", false);
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
          if (module::isBM1684Family() && dyn_cast<tpu::LutOp>(op) &&
              module::isWeight(in)) {
            continue;
          }
          if (in.getType().isa<NoneType>()) {
            continue;
          }
          elt.value = in;
          elt.type = (in.getDefiningOp() != nullptr &&
                      module::isWeight(in))
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
        if (module::isBM1684Family() && module::isWeight(ts_tensors[i].first) &&
            llvm::any_of(ts_tensors[i].first.getUsers(),
                         [](Operation *op) { return isa<tpu::LutOp>(op); })) {
          continue;
        }
        if (is_lmem_ldst(ts_tensors[i].second.mode)) {
          elt.value = ts_tensors[i].first;
          if (elt.value.getDefiningOp() &&
              module::isWeight(elt.value)) {
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

/*
  this only for Lut BM1684
*/
bool assignL2memAddr(const LgInfo &lg_info, BasicTimeStepPtr &time_step) {
  auto &l2mem_buffer = time_step->get_l2mem_buffer();
  int l2mem_start_addr = (0x22000 + 0x80000);
  int l2mem_pos = l2mem_start_addr;
  for (auto &buffer : l2mem_buffer) {
    buffer.second.addr = l2mem_pos;
    buffer.second.size = get_buffer_size(
        buffer.first.value, time_step->get_tensor_infos()[buffer.first.value],
        lg_info.type);
    l2mem_pos += buffer.second.size;
  }
  return true;
}

bool LmemAllocator::assignLmemAddr(const LgInfo &lg_info,
                                   BasicTimeStepPtr &time_step,
                                   const shape_secs_t &shape_secs,
                                   bool allow_bank_conflict) {
  PROFILE_LOG("assignLmemAddr", true);
  time_step->update_all_mem_buffer_size(lg_info);
  bool one_loop =
      (shape_secs.nsecs == 1 && shape_secs.hsecs == 1 &&
       shape_secs.csecs == 1 && shape_secs.dsecs == 1 && shape_secs.wsecs == 1);

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
    DEBUG_WITH_TYPE("assign_lmem", {
      llvm::dbgs() << "; action = assign_lmem" << "; step = initial"
                   << "; tgt_position = " << tgt_position
                   << "; lmem_occupy = " << lmem_occupy << "\n";
    });
    update_membuf_conflict_param(npu_membuf_heap, gdma_membuf_heap,
                                 membuf_list);
    membuf_list.sort(membuf_sort_std_cmp);

    for (buflist_it = membuf_list.begin(); buflist_it != membuf_list.end();
         ++buflist_it) {
      if (first_alloc) {
        first_alloc = false;
        DEBUG_WITH_TYPE("assign_lmem", {
          llvm::dbgs() << "; action = assign_lmem" << "; step = first_alloc"
                       << "; op = " << module::getName(buflist_it->first.value)
                       << "\n";
        });
        if (time_step->get_lmem_size(buflist_it->first) <= Arch::LMEM_BYTES) {
          tgt_position = 0;
          tgt_buflist_it = buflist_it;
          DEBUG_WITH_TYPE("assign_lmem", {
            llvm::dbgs() << "; action = assign_lmem"
                         << "; step = first_alloc_success"
                         << "; tgt_position = " << tgt_position
                         << "; lmem_occupy = " << lmem_occupy << "\n";
          });
        } else {
          DEBUG_WITH_TYPE("assign_lmem", {
            llvm::dbgs() << "; action = assign_lmem"
                         << "; step = find_op_assign_failed"
                         << "; tgt_position = " << tgt_position
                         << "; lmem_occupy = " << lmem_occupy << "; op = "
                         << module::getName(buflist_it->first.value) << "\n";
          });
          PROFILE_LOG("assignLmemAddr", false);
          return false;
        }
        break;
      } else {
        alloc_lmem = global_find_avail_lmem_localtion(
            buffer_avail_space[buflist_it->first], buflist_it->first,
            recent_buffer_allocated, time_step, one_loop, allow_bank_conflict);

        DEBUG_WITH_TYPE("assign_lmem", {
          llvm::dbgs() << "; action = assign_lmem"
                       << "; step = find_avail_lmem_location"
                       << "; op = " << module::getName(buflist_it->first.value)
                       << "\n";
        });
        if (alloc_lmem.first != -1) {
          if (alloc_lmem.first < tgt_position) {
            tgt_position = alloc_lmem.first;
            tgt_buflist_it = buflist_it;
            DEBUG_WITH_TYPE("assign_lmem", {
              llvm::dbgs() << "; action = assign_lmem"
                           << "; step = update_min_tgt_position"
                           << "; tgt_position = " << tgt_position
                           << "; lmem_occupy = " << lmem_occupy << "\n";
            });
          }
        } else {
          DEBUG_WITH_TYPE("assign_lmem", {
            llvm::dbgs() << "; action = assign_lmem"
                         << "; step = find_op_assign_failed" << "; op = "
                         << module::getName(buflist_it->first.value) << "\n";
          });
          PROFILE_LOG("assignLmemAddr", false);
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
      DEBUG_WITH_TYPE("assign_lmem", {
        llvm::dbgs() << "; action = assign_lmem"
                     << "; step = set_lmem_addr"
                     << "; tgt_position = " << tgt_position
                     << "; lmem_occupy = " << lmem_occupy
                     << "; buffer_end = " << buffer_end << "; op = "
                     << module::getName(tgt_buflist_it->first.value) << "\n";
      });
    } else {
      llvm::errs() << "Cannot find local memory location for memory buffers\n";
      DEBUG_WITH_TYPE("assign_lmem", {
        llvm::dbgs() << "; action = assign_lmem"
                     << "; step = op_assign_failed_in_loop_end"
                     << "; op = " << module::getName(buflist_it->first.value)
                     << "\n";
      });
      PROFILE_LOG("assignLmemAddr", false);
      return false;
    }
  }

  time_step->set_lmem_occupy(lmem_occupy);

  assignL2memAddr(lg_info, time_step);
  DEBUG_WITH_TYPE("assign_lmem", {
    llvm::dbgs() << "; action = assign_lmem"
                 << "; step = final_assign_lmem_success" << "\n";
  });
  PROFILE_LOG("assignLmemAddr", false);
  return true;
}

bool LmemAllocator::assignLmemAddrWithSecs(const LgInfo &lg_info,
                                           BasicTimeStepPtr &time_step,
                                           shape_secs_t &shape_secs,
                                           bool allow_bank_conflict) {
  std::vector<std::pair<Operation *, int>> vec_op_hsecs;
  max_shape_secs_ = get_group_max_secs(lg_info, vec_op_hsecs);
  if (!allow_bank_conflict) {
    update_data_split(time_step, lg_info, shape_secs);
    DEBUG_WITH_TYPE("shape_secs", {
      llvm::dbgs() << "; action = shape_secs" << "; step = update_data_split"
                   << "; nsecs = " << shape_secs.nsecs
                   << "; csecs = " << shape_secs.csecs
                   << "; dsecs = " << shape_secs.dsecs
                   << "; hsecs = " << shape_secs.hsecs
                   << "; wsecs = " << shape_secs.wsecs << "\n";
    });
  }

  min_total_secs_ = get_split_max_secs(time_step);
  std::vector<int64_t> group_costs;
  std::vector<shape_secs_t> shape_secs_space;
  std::shared_ptr<CycleCalculator> cycle_calculator_;
  if (module::isCV18xx()) {
    Cv18xxCycleCalculator *cyc_ptr = new Cv18xxCycleCalculator();
    cycle_calculator_ = std::shared_ptr<CycleCalculator>(cyc_ptr);
  } else {
    Bm168xCycleCalculator *cyc_ptr = new Bm168xCycleCalculator();
    cycle_calculator_ = std::shared_ptr<CycleCalculator>(cyc_ptr);
  }

  if (getenv("SC_BRUTE_FORCE")){
    sc_method_brute_force(lg_info, shape_secs, allow_bank_conflict, time_step, group_costs, shape_secs_space, cycle_calculator_);
  } else {
    sc_method_quick_search(lg_info, shape_secs, allow_bank_conflict, time_step, group_costs, shape_secs_space, cycle_calculator_);

    if (module::getCoreNum() > 1){
      sc_method_multi_core(lg_info, shape_secs, allow_bank_conflict, time_step, group_costs, shape_secs_space, cycle_calculator_);
      sc_method_multi_core_v2(lg_info, shape_secs, allow_bank_conflict, time_step, group_costs, shape_secs_space, cycle_calculator_);
      sc_method_multi_core_v3(lg_info, shape_secs, allow_bank_conflict, time_step, group_costs, shape_secs_space, cycle_calculator_);
    }
  }

  if (group_costs.empty()) {
    return false;
  }

  int64_t min_index =
      std::distance(group_costs.begin(),
                    std::min_element(group_costs.begin(), group_costs.end()));
  shape_secs = shape_secs_space[min_index];

  DEBUG_WITH_TYPE("shape_secs", {
    llvm::dbgs() << "; action = shape_secs"
                 << "; step = assign_lmem_addr_with_secs"
                 << "; choose_from = " << group_costs.size()
                 << "; min_index = " << min_index
                 << "; nsecs = " << shape_secs.nsecs
                 << "; csecs = " << shape_secs.csecs
                 << "; dsecs = " << shape_secs.dsecs
                 << "; hsecs = " << shape_secs.hsecs
                 << "; wsecs = " << shape_secs.wsecs
                 << "; min_cost = " << group_costs[min_index] << "\n";
  });

  auto status = time_step->assignTimeStep(lg_info, shape_secs, true);
  if (!status) {
    return false;
  }
  status =
        assignLmemAddr(lg_info, time_step, shape_secs, allow_bank_conflict);
  if (!status) {
    return false;
  }
  return true;
}



// best but slowest
void LmemAllocator::sc_method_brute_force(
    const LgInfo &lg_info,
    shape_secs_t &shape_secs, bool allow_bank_conflict,
    BasicTimeStepPtr &time_step, std::vector<int64_t> &group_costs,
    std::vector<shape_secs_t> &shape_secs_space,
    std::shared_ptr<CycleCalculator> cycle_calculator_) {



  for (int _n = 1; _n <= max_shape_secs_.nsecs; _n++) {
    for (int _c = 1; _c <= max_shape_secs_.csecs; _c++) {
      for (int _h = 1; _h <= max_shape_secs_.hsecs; _h++) {
        // shape_secs.nsecs = increase_nsecs(shape_secs.nsecs, max_shape_secs_.nsecs);
        // shape_secs.csecs = increase_csecs(shape_secs.csecs, max_shape_secs_.csecs);
        // assign_dhwsecs(lg_info, shape_secs, ++dhw_secs, max_shape_secs_);
        for (int _d = 1; _d <= max_shape_secs_.dsecs; _d++) {
          for (int _w = 1; _w <= max_shape_secs_.wsecs; _w++) {
            shape_secs_t cur_shape_secs = {.nsecs = _n,
                                           .hsecs = _h,
                                           .dsecs = _d,
                                           .wsecs = _w,
                                           .csecs = _c};
            if (cur_shape_secs.nsecs * cur_shape_secs.csecs *
                    cur_shape_secs.dsecs * cur_shape_secs.hsecs *
                    cur_shape_secs.wsecs <
                min_total_secs_) {
              continue;
            }

            bool status =
                time_step->assignTimeStep(lg_info, cur_shape_secs, true);
            if (status == false) {
              continue;
            }

            status = assignLmemAddr(lg_info, time_step, cur_shape_secs,
                                    allow_bank_conflict);
            if (status == false) {
              continue;
            }
            int64_t _group_cost = 0;

            #pragma omp critical(get_cycle)
            _group_cost = cycle_calculator_->getGroupCycle(
                  time_step, cur_shape_secs, lg_info.type);

            DEBUG_WITH_TYPE("shape_secs", {
              llvm::dbgs() << "; action = shape_secs"
                           << "; step = sc_method_brute_force"
                           << "; nsecs = " << cur_shape_secs.nsecs
                           << "; csecs = " << cur_shape_secs.csecs
                           << "; dsecs = " << cur_shape_secs.dsecs
                           << "; hsecs = " << cur_shape_secs.hsecs
                           << "; wsecs = " << cur_shape_secs.wsecs
                           << "; cost = " << _group_cost << "\n";
            });
            group_costs.push_back(_group_cost);
            shape_secs_space.push_back(cur_shape_secs);
          }
        }
      }
    }
  }
}

void LmemAllocator::sc_method_quick_search(
    const LgInfo &lg_info,
    shape_secs_t &_shape_secs, bool allow_bank_conflict,
    BasicTimeStepPtr &time_step, std::vector<int64_t> &group_costs,
    std::vector<shape_secs_t> &shape_secs_space,
    std::shared_ptr<CycleCalculator> cycle_calculator_) {

  shape_secs_t shape_secs = _shape_secs;
  int64_t try_num = 0;
  bool status = false;
  const int64_t MAX_TRY_NUM = 20;
  int64_t dhw_secs = shape_secs.dsecs * shape_secs.hsecs * shape_secs.wsecs;
  while (shape_secs.nsecs <= max_shape_secs_.nsecs &&
         shape_secs.dsecs <= max_shape_secs_.dsecs &&
         shape_secs.hsecs <= max_shape_secs_.hsecs &&
         shape_secs.wsecs <= max_shape_secs_.wsecs &&
         shape_secs.csecs <= max_shape_secs_.csecs) {
    // reassign time step
    status = time_step->assignTimeStep(lg_info, shape_secs, true);
    if (status == false) {
      break;
    }
    status =
        assignLmemAddr(lg_info, time_step, shape_secs, allow_bank_conflict);

    if (status == false) {
      update_shape_secs(lg_info, shape_secs, dhw_secs, max_shape_secs_);
    } else {

      int64_t _group_cost;
      #pragma omp critical(get_cycle)
      _group_cost = cycle_calculator_->getGroupCycle(
              time_step, shape_secs, lg_info.type);

      DEBUG_WITH_TYPE("shape_secs", {
        llvm::dbgs() << "; action = shape_secs"
                     << "; step = sc_method_quick_search"
                     << "; nsecs = " << shape_secs.nsecs
                     << "; csecs = " << shape_secs.csecs
                     << "; dsecs = " << shape_secs.dsecs
                     << "; hsecs = " << shape_secs.hsecs
                     << "; wsecs = " << shape_secs.wsecs
                     << "; cost = " << _group_cost
                     << "\n";
      });
      group_costs.push_back(_group_cost);
      shape_secs_space.push_back(shape_secs);
      break;
    }
    if (++try_num >= MAX_TRY_NUM) {
      break;
    }
  }
}


void LmemAllocator::sc_method_multi_core(
    const LgInfo &lg_info,
    shape_secs_t &_shape_secs, bool allow_bank_conflict,
    BasicTimeStepPtr &time_step, std::vector<int64_t> &group_costs,
    std::vector<shape_secs_t> &shape_secs_space,
    std::shared_ptr<CycleCalculator> cycle_calculator_) {

    shape_secs_t shape_secs = _shape_secs;
    auto core_num = module::getCoreNum();
    int64_t secs = shape_secs.nsecs * shape_secs.csecs * shape_secs.hsecs;
    int64_t max_secs =
        max_shape_secs_.nsecs * max_shape_secs_.csecs * max_shape_secs_.hsecs;
    if (max_secs < core_num || secs >= core_num)
      return;

    shape_secs.nsecs = max_shape_secs_.nsecs;
    secs = core_num / shape_secs.nsecs;
    if (shape_secs.csecs < secs && max_shape_secs_.csecs >= secs) {
      shape_secs.csecs = secs;
    } else if (shape_secs.csecs < secs && max_shape_secs_.csecs >= secs / 2) {
      shape_secs.csecs = secs / 2;
    }

    secs /= shape_secs.csecs;
    if (shape_secs.hsecs < secs && max_shape_secs_.hsecs >= secs) {
      shape_secs.hsecs = secs;
    } else if (shape_secs.hsecs < secs && max_shape_secs_.hsecs >= secs / 2) {
      shape_secs.hsecs = secs / 2;
    }

    //   while (1) {
    // reassign time step
    auto status = time_step->assignTimeStep(lg_info, shape_secs, true);
    if (!status) {
      return;
    }
    status =
        assignLmemAddr(lg_info, time_step, shape_secs, allow_bank_conflict);
    if (!status) {
      return;
    }

    int64_t _group_cost;
    #pragma omp critical(get_cycle)
    _group_cost = cycle_calculator_->getGroupCycle(
          time_step, shape_secs, lg_info.type);


    DEBUG_WITH_TYPE("shape_secs", {
      llvm::dbgs() << "; action = shape_secs"
                   << "; step = sc_method_multi_core"
                   << "; nsecs = " << shape_secs.nsecs
                   << "; csecs = " << shape_secs.csecs
                   << "; dsecs = " << shape_secs.dsecs
                   << "; hsecs = " << shape_secs.hsecs
                   << "; wsecs = " << shape_secs.wsecs
                   << "; cost = " << _group_cost
                   << "\n";
    });

    group_costs.push_back(_group_cost);
    shape_secs_space.push_back(shape_secs);
}





void LmemAllocator::sc_method_multi_core_v2(
    const LgInfo &lg_info,
    shape_secs_t &shape_secs, bool allow_bank_conflict,
    BasicTimeStepPtr &time_step, std::vector<int64_t> &group_costs,
    std::vector<shape_secs_t> &shape_secs_space,
    std::shared_ptr<CycleCalculator> cycle_calculator_) {

    int64_t nch_secs = shape_secs.nsecs * shape_secs.csecs * shape_secs.hsecs;
    auto core_num = module::getCoreNum();

    if (nch_secs % core_num == 0 || nch_secs < core_num){
      return;
    }

    int64_t max_nch_secs = max_shape_secs_.nsecs * max_shape_secs_.csecs * max_shape_secs_.hsecs;

    int64_t min_nch_secs =
        ceiling_func(min_total_secs_, shape_secs.wsecs * shape_secs.dsecs);

    std::vector<int64_t> history_costs;


    // search all possible n/c/h values
    for (int64_t i = min_nch_secs; i <= max_nch_secs; i++) {

      std::vector<int64_t> factorys;

      std::unordered_map<int64_t, int> counter;
      get_factory(i, factorys);
      for (auto num : factorys) {
        counter[num]++;
      }

      auto distributions = find_distributions(factorys, {max_shape_secs_.nsecs, max_shape_secs_.csecs, max_shape_secs_.hsecs});

      shape_secs_t core_shape_secs = shape_secs;
      for (const auto &dist : distributions) {
        core_shape_secs.nsecs = dist[0];
        core_shape_secs.csecs = dist[1];
        core_shape_secs.hsecs = dist[2];

        // Check if this combination is valid
        bool status = time_step->assignTimeStep(lg_info, core_shape_secs, true);

        if (status) {
          status = assignLmemAddr(lg_info, time_step, core_shape_secs);
        }

        if (status) {
          // Valid combination found, update shape_secs and return

          int64_t _group_cost;

          #pragma omp critical(get_cycle)
          _group_cost = cycle_calculator_->getGroupCycle(
              time_step, core_shape_secs, lg_info.type);

          DEBUG_WITH_TYPE("shape_secs", {
            llvm::dbgs() << "; action = shape_secs"
                        << "; step = sc_method_multi_core_v2"
                        << "; nch_secs = " << i
                        << "; nsecs = " << core_shape_secs.nsecs
                        << "; csecs = " << core_shape_secs.csecs
                        << "; dsecs = " << core_shape_secs.dsecs
                        << "; hsecs = " << core_shape_secs.hsecs
                        << "; wsecs = " << core_shape_secs.wsecs
                      << "; cost = " << _group_cost
                        << "\n";
          });
          group_costs.push_back(_group_cost);
          shape_secs_space.push_back(core_shape_secs);
        }
      }
    }
}

void LmemAllocator::sc_method_multi_core_v3(
    const LgInfo &lg_info, shape_secs_t &_shape_secs, bool allow_bank_conflict,
    BasicTimeStepPtr &time_step, std::vector<int64_t> &group_costs,
    std::vector<shape_secs_t> &shape_secs_space,
    std::shared_ptr<CycleCalculator> cycle_calculator_) {

  shape_secs_t shape_secs = _shape_secs;
  auto num_cores = module::getCoreNum();
  if (num_cores < 2) {
    return;
  }
  auto pre_secs = shape_secs.nsecs * shape_secs.csecs * shape_secs.dsecs;
  if (pre_secs * shape_secs.hsecs % num_cores == 0) {
    return;
  }
  for (int i = 1; i < num_cores; i++) {
    if ((shape_secs.hsecs + i) > max_shape_secs_.hsecs) {
      continue;
    }
    if ((pre_secs * (shape_secs.hsecs + i)) % num_cores == 0) {
      shape_secs.hsecs += i;
      auto status = time_step->assignTimeStep(lg_info, shape_secs, true);
      if (!status) {
        return;
      }
      status =
          assignLmemAddr(lg_info, time_step, shape_secs, allow_bank_conflict);
      if (!status) {
        return;
      }

      int64_t _group_cost;
      #pragma omp critical(get_cycle)
      _group_cost = cycle_calculator_->getGroupCycle(
          time_step, shape_secs, lg_info.type);

      DEBUG_WITH_TYPE("shape_secs", {
        llvm::dbgs() << "; action = shape_secs"
                     << "; step = sc_method_multi_core_v3"
                     << "; nsecs = " << shape_secs.nsecs
                     << "; csecs = " << shape_secs.csecs
                     << "; dsecs = " << shape_secs.dsecs
                     << "; hsecs = " << shape_secs.hsecs
                     << "; wsecs = " << shape_secs.wsecs
                     << "; cost = " << _group_cost
                     << "\n";
      });

      group_costs.push_back(_group_cost);
      shape_secs_space.push_back(shape_secs);
      return;
    }
  }
  return;
}

void aggressive_slice_for_multicore(LmemAllocator &lmem_allocator,
                                   const LgInfo &lg_info,
                                   BasicTimeStepPtr &time_step,
                                   shape_secs_t &ir_shape_secs,
                                   shape_secs_t lg_shape_secs)
{
  // only apply 4-core sg2380
  auto core_num = module::getCoreNum();
  if (core_num != 4 ||
      ir_shape_secs.hsecs == lg_shape_secs.hsecs)
    return;

  // if bank align causes hsecs increased, allow bank conflict to
  // improve DDR read bandwidth
  if (ir_shape_secs.hsecs == (core_num + 1) && lg_shape_secs.hsecs <= core_num) {
    lg_shape_secs.hsecs = core_num;
    auto ret = lmem_allocator.assignLmemAddrWithSecs(
        lg_info, time_step, lg_shape_secs, true);
    if (ret && (lg_shape_secs.hsecs == core_num)) {
      llvm::errs() << "Aggresive slice for multicore: " << lg_info.group_ops.size()
                   << ", nsecs: " << ir_shape_secs.nsecs
                   << ", csecs: " << ir_shape_secs.csecs
                   << ", hsecs: " << ir_shape_secs.hsecs
                   << " -> " << lg_shape_secs.hsecs
                   << ", wsecs: " << ir_shape_secs.wsecs
                   << "\n";
      ir_shape_secs.hsecs = lg_shape_secs.hsecs;
    } else {
      lmem_allocator.assignLmemAddrWithSecs(lg_info, time_step, ir_shape_secs);
    }
  }
}

/// The pass for local memory allocation
class LocalMemoryAllocationPass : public LgPass {
public:
  virtual bool run(LgPassIR *pass_ir) override {
    for (size_t i = 0; i < pass_ir->lg_infos.size(); ++i) {
      if (pass_ir->lg_infos[i].group_ops.size() > 1) {
        auto lmem_allocator = LmemAllocator();
        auto shape_secs = pass_ir->shape_secs[i];
        auto ret = lmem_allocator.assignLmemAddrWithSecs(
            pass_ir->lg_infos[i], pass_ir->time_steps[i],
            pass_ir->shape_secs[i]);

        if (ret) {
          aggressive_slice_for_multicore(lmem_allocator, pass_ir->lg_infos[i],
                                         pass_ir->time_steps[i],
                                         pass_ir->shape_secs[i],
                                         shape_secs);
        }

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

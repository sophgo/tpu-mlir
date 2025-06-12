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
  // LMEM_OPERATION is an operation buffer, so it is always used by npu
  // LMEM_ACTIVATION/LMEM_WEIGHT can be used by gdma (in store/load op), so it
  // can't return true directly
  if (buffer_key.type == LMEM_OPERATION) {
    return true;
  }
  auto users = buffer_key.value.getUsers();
  auto src_op = buffer_key.value.getDefiningOp();
  for (auto op : cur_layers) {
    if (src_op == op /* src_op is an output */ ||
        std::find(users.begin(), users.end(), op) !=
            users.end() /* src_op is an input */) {
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
        // TODO: maybe it can be move to outside in update_exclude_banks
        // function
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

static inline int64_t increase_nsecs(int64_t nsecs, int64_t max_nsecs) {
  if (nsecs == max_nsecs) {
    return -1;
  }
  int64_t nslice = max_nsecs / nsecs + (max_nsecs % nsecs > 0);
  int64_t new_nslice = nslice;
  int64_t next_nsecs = nsecs;
  do {
    next_nsecs++;
    new_nslice = max_nsecs / next_nsecs + (max_nsecs % next_nsecs > 0);
  } while (new_nslice >= nslice && next_nsecs < max_nsecs);
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
                                     const shape_secs_t &max_shape_secs,
                                     const LgOptions &options) {
  if (shape_secs.nsecs < max_shape_secs.nsecs) {
    shape_secs.nsecs = increase_nsecs(shape_secs.nsecs, max_shape_secs.nsecs);
  } else if (shape_secs.csecs < max_shape_secs.csecs) {
    shape_secs.csecs = increase_csecs(shape_secs.csecs, max_shape_secs.csecs);
  } else {
    assign_dhwsecs(lg_info, shape_secs, ++dhw_secs, max_shape_secs, options);
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
    /**
     * Case 1: full overlap
     * avail:     |--------|
     * exclude:  |----------|
     * result:    (delete)
     *
     */
    if (avail_start >= exclude_start && avail_end <= exclude_end) {
      avail_iter = avail_lmems.erase(avail_iter);
    }
    /**
     * Case 2: right overlap
     * avail:   |--------|
     * exclude:     |--------|
     * result:  |---|
     */
    else if (avail_start < exclude_start && avail_end > exclude_start &&
             avail_end <= exclude_end) {
      avail_iter->second = exclude_start - avail_start;
      avail_iter++;
    }
    /**
     * Case 3: left overlap
     * avail:       |--------|
     * exclude:  |-------|
     * result:           |---|
     */
    else if (avail_start >= exclude_start && avail_start < exclude_end &&
             avail_end > exclude_end) {
      if (avail_start == exclude_start) {
        space_split = true;
      }
      avail_iter->second = avail_end - exclude_end;
      avail_iter->first = exclude_end;
      avail_iter++;
    }
    /**
     * Case 4: full split
     * avail:   |--------------|
     * exclude:     |-----|
     * result:  |---|     |---|
     */
    else if (avail_start < exclude_start && avail_end > exclude_end) {
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
  if (op->getNumOperands() > 0) {
    auto parent = op->getOperand(0).getUsers();
    int users_num = std::distance(parent.begin(), parent.end());
    if (users_num > 1) {
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
            src_buffer_value.end_ts == buffer_value.start_ts && !need_store) {
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
    bool hold_on_coeff, bool consider_inplace, bool allow_hold_in_lmem) {
  PROFILE_LOG("update_avail_lmems", true);
  // find the allocated buffer overlap in time dimension
  bool ts_overlap = is_timestep_overlapped(
      buffer_value.start_ts, buffer_value.end_ts, recent_buffer_value.start_ts,
      recent_buffer_value.end_ts);

  if (!ts_overlap && hold_on_coeff && allow_hold_in_lmem) {
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
    const mem_buffer_value_t &buffer_value, const LgInfo &lg_info,
    bool allow_bank_conflict) {

  MemBlock alloc_lmem(-1, -1);
  if (avail_space.avail_lmems.empty()) {
    GROUP_DEBUG_WITH_TYPE("find_avail_lmem", lg_info, [&]() {
      llvm::dbgs() << LOG_STEP("avail_lmems_empty") << "\n";
    });
  } else if (allow_bank_conflict) {
    alloc_lmem = avail_space.avail_lmems.front();
    GROUP_DEBUG_WITH_TYPE("find_avail_lmem", lg_info, [&]() {
      llvm::dbgs() << LOG_STEP("use_bank_conflict_buffer")
                   << LOG_KV("lmem", alloc_lmem.first)
                   << LOG_KV("size", alloc_lmem.second) << "\n";
    });
  } else {
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
      GROUP_DEBUG_WITH_TYPE("find_avail_lmem", lg_info, [&]() {
        llvm::dbgs() << LOG_STEP("iter_avail_lmem")
                     << LOG_KV("lmem", avail_iter->first)
                     << LOG_KV("size", avail_iter->second) << "\n";
      });
      if (avail_iter->second >= buffer_value.size) {
        alloc_lmem = *avail_iter;
        GROUP_DEBUG_WITH_TYPE("find_avail_lmem", lg_info, [&]() {
          llvm::dbgs() << LOG_STEP("find_availble_buffer")
                       << LOG_KV("lmem", alloc_lmem.first)
                       << LOG_KV("size", alloc_lmem.second) << "\n";
        });
        break;
      }
    }

    // allow bank confict if could not find space not conflict
    if (alloc_lmem.first == -1) {
      alloc_lmem = avail_space.avail_lmems.front();
      GROUP_DEBUG_WITH_TYPE("find_avail_lmem", lg_info, [&]() {
        llvm::dbgs() << LOG_STEP("use_bank_conflict_buffer")
                     << LOG_KV("lmem", alloc_lmem.first)
                     << LOG_KV("size", alloc_lmem.second) << "\n";
      });
    }
  }

  if (alloc_lmem.first >= Arch::LMEM_BYTES) {
    alloc_lmem = MemBlock(-1, -1);
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
  // find in all timesteps in buffer life cycle
  for (int64_t ts = buffer_value.start_ts;
       (ts != ((buffer_value.end_ts + 1) % timestep_num)) || first_step;
       ts = (ts + 1) % timestep_num) {
    first_step = false;
    const TpuTsField &cur_layers = time_step->getLayers(ts);
    const GdmaTsField &cur_tensors = time_step->getTensors(ts);
    is_npu_use = is_buffer_used_by_npu(buffer_key, cur_layers);
    is_gdma_use = is_buffer_used_by_gdma(buffer_key, cur_tensors, is_npu_use);
    DEBUG_WITH_TYPE("assign_lmem", {
      llvm::dbgs() << LOG_ACTION("update_exclude_banks")
                   << LOG_STEP("find_banks_used") << LOG_KV("ts", ts)
                   << LOG_KV("is_npu_use", is_npu_use)
                   << LOG_KV("is_gdma_use", is_gdma_use) << "\n";
    });
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
    BasicTimeStepPtr &time_step, bool one_loop, const LgInfo &lg_info,
    bool allow_bank_conflict, bool allow_hold_in_lmem) {
  PROFILE_LOG("global_find_avail_lmem_localtion", true);
  auto &buffer_value = time_step->get_lmem_buffer_value(buffer_key);
  auto &recent_buffer_value =
      time_step->get_lmem_buffer_value(recent_buffer_allocated);

  update_exclude_banks(avail_space.exclude_banks, buffer_key, buffer_value,
                       recent_buffer_allocated, recent_buffer_value, time_step);

  update_avail_lmems(avail_space.avail_lmems, buffer_key, buffer_value,
                     recent_buffer_allocated, recent_buffer_value, time_step,
                     !one_loop, true, allow_hold_in_lmem);

  // get the available local memory location
  auto alloc_mem = find_avail_lmem_location(
      avail_space, buffer_key, buffer_value, lg_info, allow_bank_conflict);
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
          elt.type = (in.getDefiningOp() != nullptr && module::isWeight(in))
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
          if (elt.value.getDefiningOp() && module::isWeight(elt.value)) {
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
                             const BasicTimeStepPtr &time_step, bool one_loop,
                             bool allow_hold_in_lmem) {
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

    if (!one_loop && hold_on_coeff && allow_hold_in_lmem) {
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

void dump_lmem_assign_result(
    std::list<MemBufSortStd>::iterator &membuf_allocated, const LgInfo &lg_info,
    BasicTimeStepPtr &time_step, bool allow_bank_conflict,
    bool allow_hold_in_lmem, const shape_secs_t &shape_secs,
    const char *lmem_assign_status, const char *lmem_assign_remark,
    int64_t lmem_assign_addr, bool one_loop) {
  auto hold_in_lmem =
      membuf_allocated->first.type != LMEM_OPERATION &&
      time_step->is_tensor_hold_in_lmem(membuf_allocated->first.value);
  hold_in_lmem = allow_hold_in_lmem && hold_in_lmem;
  Operation *op =
      membuf_allocated->first.type == LMEM_OPERATION
          ? membuf_allocated->first.op
          : module::getOriValue(membuf_allocated->first.value).getDefiningOp();
  llvm::dbgs() << DEBUGGER_DEFAULT_INFO("dump_lmem_assign_result",
                                        "iteration_result", lmem_assign_remark)
               << LOG_KV("status", lmem_assign_status)
               << LOG_KV("lmem_type", membuf_allocated->first.lmem_type_str())
               << LOG_KV("op_type", op->getName())
               << LOG_KV("op_name",
                         module::getName(membuf_allocated->first.value));
  if (lmem_assign_addr < 0) {
    llvm::dbgs() << LOG_KV("addr", lmem_assign_addr);
  } else {
    llvm::dbgs() << LOG_KV("addr", llvm::format("0x%08x", lmem_assign_addr));
  }
  llvm::dbgs()
      << LOG_KV("size", time_step->get_lmem_size(membuf_allocated->first))
      << LOG_KV(
             "timestep_start",
             time_step->get_lmem_buffer_value(membuf_allocated->first).start_ts)
      << LOG_KV(
             "timestep_end",
             time_step->get_lmem_buffer_value(membuf_allocated->first).end_ts)
      << LOG_KV("timestep_mode",
                time_step->get_tensor_mode_str(membuf_allocated->first.value))
      << LOG_KV("hold_in_lmem", hold_in_lmem)
      << LOG_KV("allow_bank_conflict", allow_bank_conflict)
      << LOG_KV("one_loop", one_loop)
      << LOG_KV_FORMAT("shape_secs", "%d,%d,%d,%d,%d", shape_secs.nsecs,
                       shape_secs.csecs, shape_secs.dsecs, shape_secs.hsecs,
                       shape_secs.wsecs)
      << "\n";
}

void dump_membuf_list(const LgInfo &lg_info,
                      std::list<MemBufSortStd> &membuf_list, const char *step,
                      const char *tag, const char *remark) {
  GROUP_DEBUG_WITH_TYPE("membuf_list", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(step, tag, remark) << "\n";
    int i = 0;
    for (auto &iter : membuf_list) {
      GROUP_DEBUG_WITH_TYPE("membuf_list", lg_info, [&]() {
        llvm::dbgs() << LOG_KV("buf_idx", i)
                     << LOG_KV("op_conflict", iter.first.conflict)
                     << LOG_KV("op_type", iter.first.lmem_type_str())
                     << LOG_KV("start_ts", iter.second.start_ts)
                     << LOG_KV("area", iter.second.area);
        if (iter.first.type == LMEM_OPERATION) {
          llvm::dbgs() << LOG_KV("op_name", module::getName(iter.first.op));
        } else {
          llvm::dbgs() << LOG_KV("op_name", module::getName(iter.first.value));
        }
        llvm::dbgs() << "\n";
      });
      ++i;
    }
  });
}

bool LmemAllocator::assignLmemAddr(const LgInfo &lg_info,
                                   BasicTimeStepPtr &time_step,
                                   const shape_secs_t &shape_secs,
                                   bool allow_bank_conflict) {
  /**
   *
   * assignLmemAddr is function for assign lmem addr for each mem buffer defined
   * in lmem_buffer_
   *
   * all lmem_buffer_ is a map<mem_buffer_key_t, mem_buffer_value_t>
   *
   * mem_buffer_value_t is defined in BasicTimeStep.h:
   * typedef struct mem_buffer_value {
   *   int64_t start_ts;
   *   int64_t end_ts;
   *   int64_t addr;
   *   int64_t size;
   *   int64_t align_bytes;
   * } mem_buffer_value_t;
   *
   * we should fill all variable for each mem_buffer_key_t
   *
   * variable `start_ts`, `end_ts` and `align_bytes` is filled by function
   * `time_step->update_all_mem_buffer_size`
   *
   * variable `addr` and `size` is filled in this function (assignLmemAddr)
   *
   */
  PROFILE_LOG("assignLmemAddr", true);

  // iterate all mem_buffer_key_t, then update mem_buffer_value_t.size
  GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "update_all_mem_buffer_size", "call_function",
                        "update info in `lmem_buffer_`, first update "
                        "`start_ts`, `end_ts` and `align_bytes`, "
                        "then calculate `size`")
                 << "\n";
  });
  time_step->update_all_mem_buffer_size(lg_info);

  bool one_loop =
      (shape_secs.nsecs == 1 && shape_secs.hsecs == 1 &&
       shape_secs.csecs == 1 && shape_secs.dsecs == 1 && shape_secs.wsecs == 1);

  // check whether to allow tensors hold in local memory
  bool allow_hold_in_lmem = !one_loop;
  if (allow_hold_in_lmem) {
    GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "allow_hold_in_lmem_judgment", "stamp",
                          "if the size of tensors hold in local memory is "
                          "larger than local memory, "
                          "then not allow any tensors to hold in local memory")
                   << "\n";
    });
    const MemBuff &lmem_buffer = time_step->get_lmem_buffer();
    int64_t lmem_size_hold_in_lmem = 0;
    for (auto iter = lmem_buffer.begin(); iter != lmem_buffer.end(); ++iter) {
      if (iter->first.type != LMEM_OPERATION &&
          time_step->is_tensor_hold_in_lmem(iter->first.value)) {
        lmem_size_hold_in_lmem += iter->second.size;
      }
    }

    allow_hold_in_lmem = lmem_size_hold_in_lmem < Arch::LMEM_BYTES;
    if (!allow_hold_in_lmem) {
      int64_t n, c, d, h, w;
      for (size_t ts = 0; ts < time_step->get_timestep_num(); ++ts) {
        auto &cur_ts_tensors = time_step->getTensors(ts);
        for (auto &tensor : cur_ts_tensors) {
          tensor_info_t &ti = tensor.second;
          if (ti.mode == TIMESTEP_LOAD) {
            auto in = tensor.first;
            if (module::isBM1684Family() && module::isWeight(tensor.first) &&
                llvm::any_of(tensor.first.getUsers(), [](Operation *op) {
                  return isa<tpu::LutOp>(op);
                })) {
              // 1684 LutOp use l2mem for weight
              continue;
            }
            if (time_step->is_tensor_hold_in_lmem(in)) {
              GROUP_DEBUG_WITH_TYPE("lmem_buffer", lg_info, [&]() {
                llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                                    "lmem_buffer", "stamp",
                                    "cancel tensor hold in local memory")
                             << LOG_KV("name", module::getName(in)) << "\n";
              });
              time_step->cancel_tensor_hold_in_lmem(in);
              module::getNCDHW(in, n, c, d, h, w, lg_info.type);
              ti.slice_info.n.clear();
              ti.slice_info.c.clear();
              ti.slice_info.d.clear();
              ti.slice_info.h.clear();
              ti.slice_info.w.clear();
              for (int i = 0; i < shape_secs.nsecs; ++i) {
                ti.slice_info.n.push_back(
                    std::make_pair((int64_t)0, (int64_t)n));
              }
              for (int i = 0; i < shape_secs.csecs; ++i) {
                ti.slice_info.c.push_back(
                    std::make_pair((int64_t)0, (int64_t)c));
              }
              for (int i = 0; i < shape_secs.dsecs; ++i) {
                ti.slice_info.d.push_back(
                    std::make_pair((int64_t)0, (int64_t)d));
              }
              for (int i = 0; i < shape_secs.hsecs; ++i) {
                ti.slice_info.h.push_back(
                    std::make_pair((int64_t)0, (int64_t)h));
              }
              for (int i = 0; i < shape_secs.wsecs; ++i) {
                ti.slice_info.w.push_back(
                    std::make_pair((int64_t)0, (int64_t)w));
              }
            }
          }
        }
      }
    }
  }

  // init membuf_list
  std::list<MemBufSortStd> membuf_list;
  init_membuf_list(membuf_list, time_step, one_loop, allow_hold_in_lmem);

  // init avail_lmems and exclude_banks
  BufferAvailSpace buffer_avail_space;
  init_buffer_avail_space(buffer_avail_space, membuf_list);

  // create conflict heap
  std::vector<std::set<mem_buffer_key_t *>> npu_membuf_heap;
  std::vector<std::set<mem_buffer_key_t *>> gdma_membuf_heap;
  membuf_heap_create(npu_membuf_heap, gdma_membuf_heap, membuf_list, time_step);

  MemBlock candidate_allocation;
  int64_t tgt_min_address = 0;
  int64_t lmem_occupy = 0;
  bool is_first_alloc = true;
  mem_buffer_key_t recent_buffer_allocated;
  std::list<MemBufSortStd>::iterator buflist_it;
  std::list<MemBufSortStd>::iterator tgt_membuf;
  GROUP_DEBUG_WITH_TYPE("lmem_assign", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "lmem_spec", "stamp",
                        "show local memory spec of current chip")
                 << LOG_KV("lmem_eu_bytes", Arch::EU_BYTES)
                 << LOG_KV("lmem_npu_num", Arch::NPU_NUM)
                 << LOG_KV("lmem_bytes", Arch::LMEM_BYTES)
                 << LOG_KV("lmem_banks", Arch::LMEM_BANKS)
                 << LOG_KV("lmem_bank_bytes", Arch::LMEM_BANK_BYTES) << "\n";
  });

  // while loop for all membuf_list
  GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "main_loop", "stamp",
                        "while loop of `membuf_list` to allocate address for "
                        "all memory buffer")
                 << "\n";
  });
  addr_assign_result_t addr_assign_result = ADDR_ALLOCATE_SUCCESS;
  while (!membuf_list.empty()) {
    GROUP_DEBUG_WITH_TYPE("lmem_assign", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO("iteration_start", "stamp", "\\")
                   << LOG_KV("lmem_occupy", lmem_occupy)
                   << LOG_KV("membuf_need_allocate", membuf_list.size())
                   << "\n";
    });

    tgt_min_address = Arch::LMEM_BYTES;

    update_membuf_conflict_param(npu_membuf_heap, gdma_membuf_heap,
                                 membuf_list);

    // sort the membuf_list according to `conflict`, `area` and `start_ts`
    dump_membuf_list(lg_info, membuf_list, "membuf_list_before_sort", "stamp",
                     "\\");

    membuf_list.sort(membuf_sort_std_cmp);

    dump_membuf_list(lg_info, membuf_list, "membuf_list_after_sort", "stamp",
                     "\\");

    // 1. find the min address from all candidate allocations of `buflist_it`s
    // as target address
    //    and take the corresponding `buflist_it` as target memory buffer to
    //    allocate
    GROUP_DEBUG_WITH_TYPE("lmem_assign", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO("start_find_target", "stamp",
                                            "traverse all iter in membuf_list "
                                            "and find the target buffer with "
                                            "minimum address to allocate")
                   << "\n";
    });
    for (buflist_it = membuf_list.begin(); buflist_it != membuf_list.end();
         ++buflist_it) {

      // 1.1 allocate available address for current buffer iter
      if (is_first_alloc) {
        //  1.1.a first allocation can start at 0 directly
        GROUP_DEBUG_WITH_TYPE("lmem_assign", lg_info, [&]() {
          llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                              "first_allocation", "stamp",
                              "directly assign address for the first lmem "
                              "buffer in membuf_list")
                       << "\n";
        });

        candidate_allocation =
            std::make_pair(0, time_step->get_lmem_size(buflist_it->first));

        if (candidate_allocation.second > Arch::LMEM_BYTES) {
          addr_assign_result = ADDR_FIRST_ALLOCATE_FAILED;
        } else {
          tgt_min_address = 0;
          tgt_membuf = buflist_it;
        }
      } else {
        // 1.1.b update `exclude_banks` and `avail_lmems` in `avail_space` for
        // current buffer iter
        //     and find the available location through updated `avail_space` for
        //     it
        // remark: this allocation is not the final allocation result, it is
        // just a
        //         candidate allocation, we will check whether this allocation
        //         is the smallest one for all buffers in the loop
        candidate_allocation = global_find_avail_lmem_localtion(
            buffer_avail_space[buflist_it->first], buflist_it->first,
            recent_buffer_allocated, time_step, one_loop, lg_info,
            allow_bank_conflict, allow_hold_in_lmem);
        if (candidate_allocation.first == -1) {
          addr_assign_result = ADDR_CANDIDATE_ALLOCATE_FAILED;
        }
      }

      GROUP_DEBUG_WITH_TYPE("lmem_assign", lg_info, [&]() {
        llvm::dbgs()
            << DEBUGGER_DEFAULT_INFO(
                   "get_candidate_allocation", "intermediate_result",
                   "this is just a intermediate result not the target "
                   "allocation result, except for the first allocation")
            << LOG_KV("op_type", buflist_it->first.lmem_type_str())
            << LOG_KV("op_name", module::getName(buflist_it->first.value))
            << LOG_KV("is_first_alloc", is_first_alloc)
            << LOG_KV(
                   "timestep",
                   time_step->get_lmem_buffer_value(buflist_it->first).start_ts)
            << "->"
            << time_step->get_lmem_buffer_value(buflist_it->first).end_ts
            << LOG_KV("addr", llvm::format_hex(candidate_allocation.first, 8))
            << LOG_KV("size", candidate_allocation.second);
        if (candidate_allocation.first != -1) {
          std::set<int64_t> used_banks;
          find_used_banks(used_banks, candidate_allocation.first,
                          candidate_allocation.second);
          llvm::dbgs() << "; banks = ";
          const char *sep = "";
          for (auto bank : used_banks) {
            llvm::dbgs() << sep << bank;
            sep = ",";
          }
        }
        llvm::dbgs() << "\n";
      });

      // 1.2 early break if this is the first allocation or this allocation is
      // failed
      if (is_first_alloc ||
          addr_assign_result == ADDR_CANDIDATE_ALLOCATE_FAILED) {
        is_first_alloc = false;
        break;
      }

      // 1.3 update tgt_min_address and tgt_membuf if current address is smaller
      if (candidate_allocation.first < tgt_min_address) {
        tgt_min_address = candidate_allocation.first;
        tgt_membuf = buflist_it;
        GROUP_DEBUG_WITH_TYPE("lmem_assign", lg_info, [&]() {
          llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                              "update_target", "stamp",
                              "update target address and buffer iter to find "
                              "the minimum address")
                       << LOG_KV("tgt_min_address", tgt_min_address)
                       << LOG_KV("op_type", tgt_membuf->first.lmem_type_str())
                       << LOG_KV("op_name",
                                 module::getName(tgt_membuf->first.value))
                       << "\n";
        });
      }
    }

    // 2.a if find any membuf iter can't find an available lmem space, return
    // false
    if (addr_assign_result > ADDR_ALLOCATE_SUCCESS) {
      // dump all failed membuf_list
      for (auto membuf_allocated_failed = membuf_list.begin();
           membuf_allocated_failed != membuf_list.end();
           ++membuf_allocated_failed) {
        GROUP_DEBUG_WITH_TYPE("lmem_assign", lg_info, [&]() {
          dump_lmem_assign_result(membuf_allocated_failed, lg_info, time_step,
                                  allow_bank_conflict, allow_hold_in_lmem,
                                  shape_secs, "failed",
                                  "early return since appear mem buffer can't "
                                  "find available lmem space",
                                  -1, one_loop);
        });
      }
      GROUP_DEBUG_WITH_TYPE("lmem_assign", lg_info, [&]() {
        llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                            "assignLmemAddr_finish", "failed",
                            "assign lmem address for all mem buffers failed")
                     << LOG_KV("membuf_allocate_failed", membuf_list.size())
                     << LOG_KV("lmem_already_used", lmem_occupy) << "%\n";
      });
      PROFILE_LOG("assignLmemAddr", false);
      return false;
    }

    // 2.b allocate this available address for this membuf
    addr_assign_result = ADDR_ALLOCATE_SUCCESS;
    recent_buffer_allocated = tgt_membuf->first;
    time_step->set_lmem_addr(tgt_membuf->first, tgt_min_address);
    int64_t buffer_end =
        tgt_min_address + time_step->get_lmem_size(tgt_membuf->first);
    lmem_occupy = buffer_end > lmem_occupy ? buffer_end : lmem_occupy;

    // debug info
    GROUP_DEBUG_WITH_TYPE("lmem_assign", lg_info, [&]() {
      dump_lmem_assign_result(
          tgt_membuf, lg_info, time_step, allow_bank_conflict,
          allow_hold_in_lmem, shape_secs, "success",
          "allocate available address for membuf", tgt_min_address, one_loop);
    });

    conflict_heap_delete(npu_membuf_heap, gdma_membuf_heap,
                         &(tgt_membuf->first));
    membuf_list.erase(tgt_membuf);
    buffer_avail_space.erase(tgt_membuf->first);
  }

  time_step->set_lmem_occupy(lmem_occupy);
  assignL2memAddr(lg_info, time_step);

  GROUP_DEBUG_WITH_TYPE("lmem_assign", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "assignLmemAddr_finish", "success",
                        "assign lmem address for all mem buffers success")
                 << LOG_KV("total_lmem_used", lmem_occupy)
                 << LOG_KV("utilization",
                           (lmem_occupy * 100.0 / Arch::LMEM_BYTES))
                 << "%\n";
  });

  PROFILE_LOG("assignLmemAddr", false);
  return true;
}

bool LmemAllocator::assignLmemAddrWithSecs(const LgInfo &lg_info,
                                           BasicTimeStepPtr &time_step,
                                           shape_secs_t &shape_secs,
                                           bool allow_bank_conflict,
                                           bool just_check_validation) {
  auto &lg_debugger = LgDebugger::getInstance();
  std::vector<std::pair<Operation *, int>> vec_op_hsecs;
  max_shape_secs_ = get_group_max_secs(lg_info, vec_op_hsecs);
  if (!allow_bank_conflict) {
    GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO("update_data_split",
                                            "call_function",
                                            "try to find valid `shape_secs` "
                                            "for current layer group faster")
                   << "\n";
    });
    auto shape_secs_update = shape_secs;
    if (update_data_split(time_step, lg_info, shape_secs_update, options_)) {
      shape_secs = shape_secs_update;
    }
    GROUP_DEBUG_WITH_TYPE("shape_secs", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "update_data_split", "stamp",
                          "show the shape_secs after update_data_split")
                   << LOG_KV("nsecs", shape_secs.nsecs)
                   << LOG_KV("csecs", shape_secs.csecs)
                   << LOG_KV("dsecs", shape_secs.dsecs)
                   << LOG_KV("hsecs", shape_secs.hsecs)
                   << LOG_KV("wsecs", shape_secs.wsecs) << "\n";
    });
  }

  min_total_secs_ = get_split_max_secs(time_step);
  std::vector<int64_t> group_costs;
  std::vector<shape_secs_t> shape_secs_space;
  if (module::isCV18xx()) {
    Cv18xxCycleCalculator *cyc_ptr = new Cv18xxCycleCalculator();
    cycle_calculator_ = std::shared_ptr<CycleCalculator>(cyc_ptr);
  } else {
    Bm168xCycleCalculator *cyc_ptr = new Bm168xCycleCalculator();
    cycle_calculator_ = std::shared_ptr<CycleCalculator>(cyc_ptr);
  }

  bool use_brute_force = false;
  if (lg_debugger.get_sc_method() == SC_BRUTE_FORCE &&
      lg_debugger.is_conditional_debug_group(lg_info.func_start_idx,
                                             lg_info.func_end_idx)) {
    use_brute_force = true;
  }

  if (getenv("SC_BRUTE_FORCE") || use_brute_force) {
    GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "sc_method_brute_force", "call_function",
                          "this method try all `shape_secs` for debug")
                   << "\n";
    });
    sc_method_brute_force(lg_info, shape_secs, allow_bank_conflict, time_step);
  } else if (lg_info.use_cache && lg_info.shape_secs.nsecs != 0 &&
             !getenv("RESEARCH_SHAPE_SECS")) {
    GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO("use_layergroup_cache", "stamp",
                                            "use cache info of `group_cost` "
                                            "and `shape_secs` to skip search")
                   << "\n";
    });
    min_group_costs_ = lg_info.group_cost;
    min_shape_secs_ = lg_info.shape_secs;
  } else {
    GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
      llvm::dbgs()
          << DEBUGGER_DEFAULT_INFO(
                 "sc_method_quick_search", "call_function",
                 "this method try to search valid `shape_secs` quickly")
          << "\n";
    });
    sc_method_quick_search(lg_info, shape_secs, allow_bank_conflict, time_step);

    if (module::getCoreNum() > 1) {
      sc_method_multi_core(lg_info, shape_secs, allow_bank_conflict, time_step);
      sc_method_multi_core_v2(lg_info, shape_secs, allow_bank_conflict,
                              time_step);
      sc_method_multi_core_v3(lg_info, shape_secs, allow_bank_conflict,
                              time_step);
    }
  }

  if (min_group_costs_ == -1) {
    GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO("assignLmemAddrWithSecs_finish",
                                            "failed",
                                            "assign lmem addr with secs failed")
                   << "\n";
    });
    return false;
  }

  shape_secs = min_shape_secs_;

  if (just_check_validation) {
    // do not need to re-assign lmem addrs a.
    return true;
  }

  GROUP_DEBUG_WITH_TYPE("shape_secs", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "find_valid_shape_secs", "stamp",
                        "show the valid shape_secs after searching")
                 << LOG_KV("nsecs", shape_secs.nsecs)
                 << LOG_KV("csecs", shape_secs.csecs)
                 << LOG_KV("dsecs", shape_secs.dsecs)
                 << LOG_KV("hsecs", shape_secs.hsecs)
                 << LOG_KV("wsecs", shape_secs.wsecs)
                 << LOG_KV("min_cost", min_group_costs_) << "\n";
  });

  auto status = time_step->assignTimeStep(lg_info, shape_secs, true);
  if (!status) {
    GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO("assignLmemAddrWithSecs_finish",
                                            "failed",
                                            "check `assignTimeStep` for valid "
                                            "`shape_secs` searched failed")
                   << "\n";
    });
    return false;
  }
  lg_debugger.set_do_debug(false);
  status = assignLmemAddr(lg_info, time_step, shape_secs, allow_bank_conflict);
  lg_debugger.set_do_debug(true);
  if (!status) {
    GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO("assignLmemAddrWithSecs_finish",
                                            "failed",
                                            "check `assignLmemAddr` for valid "
                                            "`shape_secs` searched failed")
                   << "\n";
    });
    return false;
  }

  GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO("assignLmemAddrWithSecs_finish",
                                          "success",
                                          "assign lmem addr with secs success")
                 << LOG_KV("min_group_costs", min_group_costs_)
                 << LOG_KV("shape_secs",
                           llvm::format("%d,%d,%d,%d,%d", shape_secs.nsecs,
                                        shape_secs.csecs, shape_secs.dsecs,
                                        shape_secs.hsecs, shape_secs.wsecs))
                 << "\n";
  });
  return true;
}

search_result_t LmemAllocator::try_this_shape_secs(
    const LgInfo &lg_info, shape_secs_t &shape_secs, bool allow_bank_conflict,
    BasicTimeStepPtr &time_step) {
  GROUP_DEBUG_WITH_TYPE("shape_secs", lg_info, [&]() {
    llvm::dbgs()
        << DEBUGGER_DEFAULT_INFO(
               "try_this_shape_secs", "stamp",
               "show current shape_secs for assignTimeStep and assignLmemAddr")
        << LOG_KV("nsecs", shape_secs.nsecs)
        << LOG_KV("csecs", shape_secs.csecs)
        << LOG_KV("dsecs", shape_secs.dsecs)
        << LOG_KV("hsecs", shape_secs.hsecs)
        << LOG_KV("wsecs", shape_secs.wsecs) << "\n";
  });
  GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
    llvm::dbgs()
        << DEBUGGER_DEFAULT_INFO(
               "assignTimeStep", "call_function",
               "backward `slice_info` of tensors starting from output tensors "
               "according to `this shape_secs`; "
               "assign and optimize timesteps for gdma and tpu operations")
        << "\n";
  });
  bool status = time_step->assignTimeStep(lg_info, shape_secs, true);

  if (!status) {
    return SECS_TIMESTEP_INVALID;
  }
  GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
    llvm::dbgs()
        << DEBUGGER_DEFAULT_INFO(
               "assignLmemAddr", "call_function",
               "try to assign local memory address under `this shape_secs`")
        << "\n";
  });
  status = assignLmemAddr(lg_info, time_step, shape_secs, allow_bank_conflict);

  if (!status) {
    return SECS_LMEM_INVALID;
  }

  int64_t _group_cost;

#pragma omp critical(get_cycle)
  _group_cost =
      cycle_calculator_->getGroupCycle(time_step, shape_secs, lg_info.type);

  last_group_cost_ = _group_cost;
  if (min_group_costs_ == -1 || _group_cost < min_group_costs_) {
    min_group_costs_ = _group_cost;
    min_shape_secs_ = shape_secs;
    return SECS_VALID_AND_BETTER;
  }

  return SECS_VALID;
}

// best but slowest
void LmemAllocator::sc_method_brute_force(const LgInfo &lg_info,
                                          shape_secs_t &shape_secs,
                                          bool allow_bank_conflict,
                                          BasicTimeStepPtr &time_step) {
  GROUP_DEBUG_WITH_TYPE("shape_secs", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "sc_method_brute_force", "stamp",
                        "show max shape_secs for brute force search")
                 << LOG_KV("max_nsecs", max_shape_secs_.nsecs)
                 << LOG_KV("max_csecs", max_shape_secs_.csecs)
                 << LOG_KV("max_dsecs", max_shape_secs_.dsecs)
                 << LOG_KV("max_hsecs", max_shape_secs_.hsecs)
                 << LOG_KV("max_wsecs", max_shape_secs_.wsecs) << "\n";
  });
  for (int _n = 1; _n <= max_shape_secs_.nsecs; _n++) {
    for (int _c = 1; _c <= max_shape_secs_.csecs; _c++) {
      for (int _h = 1; _h <= max_shape_secs_.hsecs; _h++) {
        // shape_secs.nsecs = increase_nsecs(shape_secs.nsecs,
        // max_shape_secs_.nsecs); shape_secs.csecs =
        // increase_csecs(shape_secs.csecs, max_shape_secs_.csecs);
        // assign_dhwsecs(lg_info, shape_secs, ++dhw_secs, max_shape_secs_);
        for (int _d = 1; _d <= max_shape_secs_.dsecs; _d++) {
          for (int _w = 1; _w <= max_shape_secs_.wsecs; _w++) {
            shape_secs_t cur_shape_secs;
            cur_shape_secs.nsecs = _n;
            cur_shape_secs.hsecs = _h;
            cur_shape_secs.dsecs = _d;
            cur_shape_secs.wsecs = _w;
            cur_shape_secs.csecs = _c;
            if (cur_shape_secs.nsecs * cur_shape_secs.csecs *
                    cur_shape_secs.dsecs * cur_shape_secs.hsecs *
                    cur_shape_secs.wsecs <
                min_total_secs_) {
              continue;
            }
            search_result_t ret = try_this_shape_secs(
                lg_info, cur_shape_secs, allow_bank_conflict, time_step);
            if (ret >= SECS_VALID) {
              GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
                llvm::dbgs()
                    << DEBUGGER_DEFAULT_INFO("try_this_shape_secs", "result",
                                             "show brute force search result")
                    << LOG_KV_FORMAT("shape_secs", "%d,%d,%d,%d,%d",
                                     shape_secs.nsecs, shape_secs.csecs,
                                     shape_secs.dsecs, shape_secs.hsecs,
                                     shape_secs.wsecs)
                    << LOG_KV("search_result", search_result_to_string(ret))
                    << LOG_KV("group_cost", last_group_cost_) << "\n";
              });
            }
          }
        }
      }
    }
  }
}

void LmemAllocator::sc_method_quick_search(const LgInfo &lg_info,
                                           shape_secs_t &_shape_secs,
                                           bool allow_bank_conflict,
                                           BasicTimeStepPtr &time_step) {
  GROUP_DEBUG_WITH_TYPE("shape_secs", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "sc_method_quick_search", "stamp",
                        "compare initial `shape_secs` with `max_shape_secs_` "
                        "in `sc_method_quick_search`")
                 << LOG_KV_FORMAT("init_nsecs/max_nsecs", "%d/%d",
                                  _shape_secs.nsecs, max_shape_secs_.nsecs)
                 << LOG_KV_FORMAT("init_csecs/max_csecs", "%d/%d",
                                  _shape_secs.csecs, max_shape_secs_.csecs)
                 << LOG_KV_FORMAT("init_dsecs/max_dsecs", "%d/%d",
                                  _shape_secs.dsecs, max_shape_secs_.dsecs)
                 << LOG_KV_FORMAT("init_hsecs/max_hsecs", "%d/%d",
                                  _shape_secs.hsecs, max_shape_secs_.hsecs)
                 << LOG_KV_FORMAT("init_wsecs/max_wsecs", "%d/%d",
                                  _shape_secs.wsecs, max_shape_secs_.wsecs)
                 << "\n";
  });
  shape_secs_t shape_secs = _shape_secs;
  int64_t try_num = 0;
  const int64_t MAX_TRY_NUM = 20;
  int64_t dhw_secs = shape_secs.dsecs * shape_secs.hsecs * shape_secs.wsecs;
  GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
    llvm::dbgs()
        << DEBUGGER_DEFAULT_INFO(
               "main_loop", "stamp",
               "uisng while loop try to find `shape_secs` for %d times",
               MAX_TRY_NUM)
        << "\n";
  });
  search_result_t ret;
  while (shape_secs.nsecs <= max_shape_secs_.nsecs &&
         shape_secs.dsecs <= max_shape_secs_.dsecs &&
         shape_secs.hsecs <= max_shape_secs_.hsecs &&
         shape_secs.wsecs <= max_shape_secs_.wsecs &&
         shape_secs.csecs <= max_shape_secs_.csecs) {
    // reassign time step
    GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO("iteration_start", "stamp",
                                            "already try %d times", try_num)
                   << "\n";
    });
    ret = try_this_shape_secs(lg_info, shape_secs, allow_bank_conflict,
                              time_step);
    auto ret_str = search_result_to_string(ret);
    GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO("try_this_shape_secs", "result",
                                            "already try %d times", try_num + 1)
                   << LOG_KV_FORMAT("shape_secs", "%d,%d,%d,%d,%d",
                                    shape_secs.nsecs, shape_secs.csecs,
                                    shape_secs.dsecs, shape_secs.hsecs,
                                    shape_secs.wsecs)
                   << LOG_KV("search_result", ret_str)
                   << LOG_KV("group_cost", last_group_cost_) << "\n";
    });
    if (ret >= SECS_VALID) {
      break;
    } else if (ret > SECS_TIMESTEP_INVALID) {
      update_shape_secs(lg_info, shape_secs, dhw_secs, max_shape_secs_,
                        options_);
    } else {
      break;
    }
    if (++try_num >= MAX_TRY_NUM) {
      break;
    }
  }
}

void LmemAllocator::sc_method_multi_core(const LgInfo &lg_info,
                                         shape_secs_t &_shape_secs,
                                         bool allow_bank_conflict,
                                         BasicTimeStepPtr &time_step) {

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
  status = assignLmemAddr(lg_info, time_step, shape_secs, allow_bank_conflict);
  if (!status) {
    return;
  }

  search_result_t ret =
      try_this_shape_secs(lg_info, shape_secs, allow_bank_conflict, time_step);
  if (ret >= SECS_VALID) {
    DEBUG_WITH_TYPE("shape_secs", {
      llvm::dbgs() << LOG_ACTION("shape_secs")
                   << LOG_STEP("sc_method_multi_core")
                   << LOG_KV("nsecs", shape_secs.nsecs)
                   << LOG_KV("csecs", shape_secs.csecs)
                   << LOG_KV("dsecs", shape_secs.dsecs)
                   << LOG_KV("hsecs", shape_secs.hsecs)
                   << LOG_KV("wsecs", shape_secs.wsecs)
                   << LOG_KV("cost", last_group_cost_) << "\n";
    });
  }
}

void LmemAllocator::sc_method_multi_core_v2(const LgInfo &lg_info,
                                            shape_secs_t &shape_secs,
                                            bool allow_bank_conflict,
                                            BasicTimeStepPtr &time_step) {

  int64_t nch_secs = shape_secs.nsecs * shape_secs.csecs * shape_secs.hsecs;
  auto core_num = module::getCoreNum();

  if (nch_secs % core_num == 0 || nch_secs < core_num) {
    return;
  }
  int MAX_TRY_NUM = getenv("MAX_TRY_NUM") ? atoi(getenv("MAX_TRY_NUM")) : 200;
  int64_t max_nch_secs =
      max_shape_secs_.nsecs * max_shape_secs_.csecs * max_shape_secs_.hsecs;

  int64_t min_nch_secs =
      ceiling_func(min_total_secs_, shape_secs.wsecs * shape_secs.dsecs);

  std::vector<int64_t> history_costs;

  int not_best_count = 0;
  // search all possible n/c/h values
  for (int64_t i = min_nch_secs; i <= max_nch_secs; i++) {

    std::vector<int64_t> factorys;

    std::unordered_map<int64_t, int> counter;
    get_factory(i, factorys);
    for (auto num : factorys) {
      counter[num]++;
    }

    auto distributions = find_distributions(
        factorys,
        {max_shape_secs_.nsecs, max_shape_secs_.csecs, max_shape_secs_.hsecs});

    shape_secs_t core_shape_secs = shape_secs;
    for (const auto &dist : distributions) {
      core_shape_secs.nsecs = dist[0];
      core_shape_secs.csecs = dist[1];
      core_shape_secs.hsecs = dist[2];

      search_result_t ret = try_this_shape_secs(lg_info, core_shape_secs,
                                                allow_bank_conflict, time_step);
      if (ret == SECS_VALID_AND_BETTER) {
        not_best_count = 0;
      } else if (ret == SECS_VALID) {
        // only count valid but not best
        not_best_count++;
      }

      if (ret >= SECS_VALID) {
        DEBUG_WITH_TYPE("shape_secs", {
          llvm::dbgs() << LOG_ACTION("shape_secs")
                       << LOG_STEP("sc_method_multi_core_v2")
                       << LOG_KV("nch_secs", i)
                       << LOG_KV("nsecs", core_shape_secs.nsecs)
                       << LOG_KV("csecs", core_shape_secs.csecs)
                       << LOG_KV("dsecs", core_shape_secs.dsecs)
                       << LOG_KV("hsecs", core_shape_secs.hsecs)
                       << LOG_KV("wsecs", core_shape_secs.wsecs)
                       << LOG_KV("cost", last_group_cost_) << "\n";
        });
      }
      if (not_best_count >= MAX_TRY_NUM) {
        // max search times
        return;
      }
    }
  }
}

void LmemAllocator::sc_method_multi_core_v3(const LgInfo &lg_info,
                                            shape_secs_t &_shape_secs,
                                            bool allow_bank_conflict,
                                            BasicTimeStepPtr &time_step) {

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

      search_result_t ret = try_this_shape_secs(lg_info, shape_secs,
                                                allow_bank_conflict, time_step);
      if (ret >= SECS_VALID) {
        DEBUG_WITH_TYPE("shape_secs", {
          llvm::dbgs() << LOG_ACTION("shape_secs")
                       << LOG_STEP("sc_method_multi_core_v3")
                       << LOG_KV("nsecs", shape_secs.nsecs)
                       << LOG_KV("csecs", shape_secs.csecs)
                       << LOG_KV("dsecs", shape_secs.dsecs)
                       << LOG_KV("hsecs", shape_secs.hsecs)
                       << LOG_KV("wsecs", shape_secs.wsecs)
                       << LOG_KV("cost", last_group_cost_) << "\n";
        });
      }
    }
  }
  return;
}

void aggressive_slice_for_multicore(LmemAllocator &lmem_allocator,
                                    const LgInfo &lg_info,
                                    BasicTimeStepPtr &time_step,
                                    shape_secs_t &ir_shape_secs,
                                    shape_secs_t lg_shape_secs) {
  // only apply 4-core sg2380
  auto core_num = module::getCoreNum();
  if (core_num != 4 || ir_shape_secs.hsecs == lg_shape_secs.hsecs)
    return;

  // if bank align causes hsecs increased, allow bank conflict to
  // improve DDR read bandwidth
  if (ir_shape_secs.hsecs == (core_num + 1) &&
      lg_shape_secs.hsecs <= core_num) {
    lg_shape_secs.hsecs = core_num;
    auto ret = lmem_allocator.assignLmemAddrWithSecs(lg_info, time_step,
                                                     lg_shape_secs, true);
    if (ret && (lg_shape_secs.hsecs == core_num)) {
      llvm::errs() << "Aggresive slice for multicore: "
                   << lg_info.group_ops.size()
                   << ", nsecs: " << ir_shape_secs.nsecs
                   << ", csecs: " << ir_shape_secs.csecs
                   << ", hsecs: " << ir_shape_secs.hsecs << " -> "
                   << lg_shape_secs.hsecs << ", wsecs: " << ir_shape_secs.wsecs
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
  LocalMemoryAllocationPass(const LgOptions &options) { options_ = options; }
  virtual bool run(LgPassIR *pass_ir) override {
    for (size_t i = 0; i < pass_ir->lg_infos.size(); ++i) {
      if (pass_ir->lg_infos[i].group_ops.size() > 1) {
        auto lmem_allocator = LmemAllocator(options_);
        auto shape_secs = pass_ir->shape_secs[i];
        auto ret = lmem_allocator.assignLmemAddrWithSecs(
            pass_ir->lg_infos[i], pass_ir->time_steps[i],
            pass_ir->shape_secs[i]);

        if (ret) {
          aggressive_slice_for_multicore(lmem_allocator, pass_ir->lg_infos[i],
                                         pass_ir->time_steps[i],
                                         pass_ir->shape_secs[i], shape_secs);
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

std::unique_ptr<LgPass>
CreateLocalMemoryAllocationPass(const LgOptions &options) {
  return std::unique_ptr<LgPass>(new LocalMemoryAllocationPass(options));
}

} // namespace tpu
} // namespace tpu_mlir

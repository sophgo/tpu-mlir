//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Support/LLVM.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include <list>
#include <map>
#include <set>
#include <vector>

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/BasicTimeStep.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgPass.h"

namespace tpu_mlir {
namespace tpu {

typedef struct membuf_sort_standard {
  int64_t area;
  int64_t start_ts;
} membuf_sort_std_t;

using MemBufSortStd = std::pair<mem_buffer_key_t, membuf_sort_std_t>;

typedef struct {
  std::list<MemBlock> avail_lmems;
  std::set<int64_t> exclude_banks;
} avail_space_t;
using BufferAvailSpace = std::map<mem_buffer_key_t, avail_space_t>;

class LmemAllocator {
public:
  LmemAllocator() {}
  bool assignLmemAddrWithSecs(const LgInfo &lg_info,
                              BasicTimeStepPtr &time_step,
                              shape_secs_t &shape_secs,
                              bool allow_bank_conflict = false);
  bool assignLmemAddr(const LgInfo &lg_info, BasicTimeStepPtr &time_step,
                      const shape_secs_t &shape_secs,
                      bool allow_bank_conflict = false);

  void find_used_banks(std::set<int64_t> &used_banks, int64_t local_addr,
                       int64_t local_size);

  void update_exclude_banks(std::set<int64_t> &exclude_banks,
                            const mem_buffer_key_t &buffer_key,
                            const mem_buffer_value_t &buffer_value,
                            const mem_buffer_key_t &recent_buffer_allocated,
                            const mem_buffer_value_t &recent_buffer_value,
                            BasicTimeStepPtr &time_step);

  bool update_avail_lmems(std::list<MemBlock> &avail_lmems,
                          const MemBlock &exclude_lmem);
  void update_avail_lmems(std::list<MemBlock> &avail_lmems,
                          const mem_buffer_key_t &buffer_key,
                          const mem_buffer_value_t &buffer_value,
                          const mem_buffer_key_t &recent_buffer_allocated,
                          const mem_buffer_value_t &recent_buffer_value,
                          BasicTimeStepPtr &time_step, bool hold_on_coeff,
                          bool consider_inplace);

  MemBlock find_avail_lmem_location(avail_space_t &avail_space,
                                    const mem_buffer_key_t &buffer_key,
                                    const mem_buffer_value_t &buffer_value,
                                    bool allow_bank_conflit = false);

  MemBlock global_find_avail_lmem_localtion(
      avail_space_t &avail_space, const mem_buffer_key_t &buffer_key,
      const mem_buffer_key_t &recent_buffer_allocated,
      BasicTimeStepPtr &time_step, bool one_loop,
      bool allow_bank_conflict = false);

protected:
  bool consider_inplace_;
  shape_secs_t max_shape_secs_;
  int64_t min_total_secs_;
  void sc_method_brute_force(const LgInfo &lg_info,
                             shape_secs_t &shape_secs,
                             bool allow_bank_conflict,
                             BasicTimeStepPtr &time_step,
                             std::vector<int64_t> &group_costs,
                             std::vector<shape_secs_t> &shape_secs_space,
                             std::shared_ptr<CycleCalculator> cycle_calculator_);
  void sc_method_quick_search(const LgInfo &lg_info,
                             shape_secs_t &shape_secs,
                             bool allow_bank_conflict,
                             BasicTimeStepPtr &time_step,
                             std::vector<int64_t> &group_costs,
                             std::vector<shape_secs_t> &shape_secs_space,
                             std::shared_ptr<CycleCalculator> cycle_calculator_);
  void sc_method_multi_core(const LgInfo &lg_info,
                             shape_secs_t &shape_secs,
                             bool allow_bank_conflict,
                             BasicTimeStepPtr &time_step,
                             std::vector<int64_t> &group_costs,
                             std::vector<shape_secs_t> &shape_secs_space,
                             std::shared_ptr<CycleCalculator> cycle_calculator_);
  void sc_method_multi_core_v2(const LgInfo &lg_info,
                             shape_secs_t &shape_secs,
                             bool allow_bank_conflict,
                             BasicTimeStepPtr &time_step,
                             std::vector<int64_t> &group_costs,
                             std::vector<shape_secs_t> &shape_secs_space,
                             std::shared_ptr<CycleCalculator> cycle_calculator_);
  void sc_method_multi_core_v3(const LgInfo &lg_info,
                             shape_secs_t &shape_secs,
                             bool allow_bank_conflict,
                             BasicTimeStepPtr &time_step,
                             std::vector<int64_t> &group_costs,
                             std::vector<shape_secs_t> &shape_secs_space,
                             std::shared_ptr<CycleCalculator> cycle_calculator_);
  // mem_buffer_key_t recent_buffer_allocated_;
  // std::list<std::pair<int64_t, int64_t>> avail_lmems_;
};

std::unique_ptr<LgPass> CreateLocalMemoryAllocationPass();

} // namespace tpu
} // namespace tpu_mlir

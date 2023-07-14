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

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/BasicTimeStep.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/CycleCalculator.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LmemAllocator.h"

namespace tpu_mlir {
namespace tpu {

typedef struct {
  int min_cost;
  shape_secs_t left_shape_secs;
  shape_secs_t right_shape_secs;
} SequenceGroupsInfo;

class GroupMethod {
public:
  GroupMethod(int64_t opt);
  void process(std::vector<LgInfo> &lg_infos,
               const SetVector<Operation *> &subnet_ops);
  void simple_layer_group(std::vector<LgInfo> &lg_infos,
                          const SetVector<Operation *> &subnet_ops);
  void dynamic_programming_layer_group_with_cluster(
      std::vector<LgInfo> &lg_infos, const SetVector<Operation *> &subnet_ops);

  void
  get_final_groups(std::vector<LgInfo> &lg_infos,
                   const std::vector<std::vector<Operation *>> &base_groups);

  void get_base_groups(std::vector<std::vector<Operation *>> &base_groups,
                       const SetVector<Operation *> &subnet_ops);

  int64_t get_max_cluster_size(int64_t layer_num);
  void get_group_clusters(std::vector<std::pair<int64_t, int64_t>> &clusters,
                          const std::vector<Operation *> &base_group);

  bool is_layer_group_valid(LgInfo &lg_info, bool calc_cost,
                            int64_t *group_cost);
  bool group_one_layer_proc(const LgInfo &lg_info, bool calc_cost,
                            int64_t *group_cost);

  bool group_valid_pre_check(const LgInfo &lg_info);

  bool dynamic_group_valid_check(const LgInfo &lg_info);

  bool isLgSupport(Operation *op);

  int64_t cost_add(int64_t cost0, int64_t cost1);

  void sweep_for_min_cost(int64_t *group_cost, int64_t *optimal_point,
                          int64_t start, int64_t end,
                          const std::vector<std::vector<int64_t>> &cost_table);
  void
  get_layer_cut_result(std::vector<int64_t> &cut_result,
                       const std::vector<std::pair<int64_t, int64_t>> &clusters,
                       const std::vector<std::vector<int64_t>> &cut_points,
                       int64_t start, int64_t end);

  bool update_sequence_group_cost(LgInfo *left_layer_group,
                                  LgInfo *right_layer_group, bool *left_first,
                                  SequenceGroupsInfo &seq_info,
                                  const SetVector<Operation *> &subnet_ops);
  bool merge_cut_idx_to_reduce_gdma_cost(
      const std::vector<std::vector<Operation *>> &base_groups,
      const SetVector<Operation *> &subnet_ops);
  bool consider_redundant_computation_and_gdma_cost(
      const std::vector<std::vector<Operation *>> &base_groups,
      const SetVector<Operation *> &subnet_ops);

  void show_cut_results();

protected:
  BasicTimeStepPtr time_step_;
  std::shared_ptr<LmemAllocator> lmem_allocator_;
  std::shared_ptr<CycleCalculator> cycle_calculator_;
  std::vector<std::vector<int64_t>> cut_results_;
  int64_t group_cost_;
  int64_t MAX_COST;
  int64_t opt_;
  RunMode runmode_;
};

std::unique_ptr<LgPass> CreateLayerGroupSearchPass(const LgOptions &options);

} // namespace tpu
} // namespace tpu_mlir

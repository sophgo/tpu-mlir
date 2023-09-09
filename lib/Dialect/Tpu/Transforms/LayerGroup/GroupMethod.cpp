//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "progressbar.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupMethod.h"
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "layer-group"
using namespace tpu_mlir::backend;

namespace tpu_mlir {
namespace tpu {
#define MAX_GROUP_CLUSTER (50)

#define GROUP_CHECK_RETURN(val)                                                \
  {                                                                            \
    if (val) {                                                                 \
      llvm::errs() << "layer group is valid";                                  \
      return true;                                                             \
    } else {                                                                   \
      llvm::errs() << "layer group is invalid";                                \
      return false;                                                            \
    }                                                                          \
  }

// set GROUP_3D if there is 3DOp
static bool can_be_group_3d(std::vector<Operation *> &group_ops) {
  for (auto op : group_ops) {
    if (isa<Conv3DOp, Pool3DOp>(op)) {
      return true;
    }
  }
  return false;
}

// set GROUP_NORMAL if not all ops should meet the conditions
// 1. op is eltwise-op or only the last dim cannot split
// 2. C is too small to fully utilize NPU and H is better
//    or N*C*H could be divided by NPU_NUM
static bool can_be_group_small_c(std::vector<Operation *> &group_ops) {
  auto ranmode = getRunMode(group_ops[0]);
  if (ranmode == RunMode::TPU_DYNAMIC) {
    return false;
  }
  for (auto op : group_ops) {
    if (!isa<ActiveOp, AddOp, CastOp, LayerNormOp, MulConstOp, MatMulOp,
             SoftmaxOp, RMSNormOp>(op)) {
      return false;
    }
    auto shape = module::getShape(op->getOperand(0));
    if (auto op_ = dyn_cast<LayerNormOp>(op)) {
      if (op_.getAxis() != shape.size() - 1) {
        return false;
      }
    } else if (isa<AddOp>(op)) {
      auto shapeB = module::getShape(op->getOperand(1));
      if (shape != shapeB) {
        return false;
      }
    } else if (auto op_ = dyn_cast<SoftmaxOp>(op)) {
      if (op_.getAxis() != shape.size() - 1) {
        return false;
      }
    } else if (auto op_ = dyn_cast<MatMulOp>(op)) {
      auto hdim_is_batch = op_.getHdimIsBatch();
      if (hdim_is_batch) {
        return false;
      }
    }

    if ((shape.size() == 4 &&
         shape[0] * shape[1] * shape[2] % Arch::NPU_NUM == 0) ||
        (shape.size() == 5 &&
         shape[0] * shape[1] * shape[2] * shape[3] % Arch::NPU_NUM == 0)) {
      continue;
    }

    if (!(((shape.size() == 5 && shape[3] > shape[1]) ||
           (shape.size() == 4 && shape[2] > shape[1])) &&
          shape[1] < Arch::NPU_NUM / 2)) {
      return false;
    }
  }
  return true;
}

static bool can_be_group_mm(std::vector<Operation *> &group_ops) {
  for (auto op : group_ops) {
    if (!isa<ActiveOp, AddOp, CastOp, LayerNormOp, MulConstOp, MatMulOp, MulOp,
             ReshapeOp, SoftmaxOp, AttentionOp, RMSNormOp>(op)) {
      return false;
    }
    auto shape = module::getShape(op->getOperand(0));
    if (auto op_ = dyn_cast<LayerNormOp>(op)) {
      if (op_.getAxis() != shape.size() - 1) {
        return false;
      }
    } else if (isa<AddOp, MulOp>(op)) {
      auto shapeB = module::getShape(op->getOperand(1));
      if (shape != shapeB) {
        return false;
      }
    } else if (auto op_ = dyn_cast<ReshapeOp>(op)) {
      auto ishape = module::getShape(op_.getInput());
      auto oshape = module::getShape(op_.getOutput());
      if (!(ishape.size() == 3 && oshape.size() == 4 &&
            ishape[2] == oshape[2] * oshape[3]) &&
          !(ishape.size() == 4 && oshape.size() == 3 &&
            oshape[2] == ishape[2] * ishape[3])) {
        return false;
      }
    } else if (auto op_ = dyn_cast<SoftmaxOp>(op)) {
      if (op_.getAxis() != shape.size() - 1) {
        return false;
      }
    } else if (auto op_ = dyn_cast<MatMulOp>(op)) {
      auto left_trans = op_.getLeftTranspose();
      auto right_trans = op_.getRightTranspose();
      if (left_trans && right_trans) {
        return false;
      }
    } else if (auto op_ = dyn_cast<AttentionOp>(op)) {
      if (module::isNone(op_.getKeys())) {
        return false;
      }
    }
  }

  return true;
}

static void set_group_type(LgInfo &lg_info) {
  lg_info.type = GROUP_NORMAL;
  if (lg_info.group_ops.size() == 1) {
    return;
  }

  if (can_be_group_3d(lg_info.group_ops)) {
    lg_info.type = GROUP_3D;
    return;
  }

  if (module::isCV18xx() || module::isBM1684Family()) {
    // cv18xx only support GROUP_NORMAL
    lg_info.type = GROUP_NORMAL;
    return;
  }

  if (can_be_group_small_c(lg_info.group_ops)) {
    lg_info.type = GROUP_SMALL_C;
    return;
  }

  if (can_be_group_mm(lg_info.group_ops)) {
    lg_info.type = GROUP_MM;
    return;
  }
}

static void get_layer_group(LgInfo &lg_info,
                            const std::vector<Operation *> &base_group,
                            int64_t left, int64_t right) {
  lg_info.clear();
  for (int idx = left; idx <= right; ++idx) {
    lg_info.group_ops.push_back(base_group[idx]);
  }
  lg_info.update_group_io();
  set_group_type(lg_info);
}

GroupMethod::GroupMethod(int64_t opt) {
  if (module::isCV18xx()) {
    Cv18xxCycleCalculator *cyc_ptr = new Cv18xxCycleCalculator();
    cycle_calculator_ = std::shared_ptr<CycleCalculator>(cyc_ptr);
  } else {
    Bm168xCycleCalculator *cyc_ptr = new Bm168xCycleCalculator();
    cycle_calculator_ = std::shared_ptr<CycleCalculator>(cyc_ptr);
  }
  MAX_COST = llvm::maxIntN(64);
  opt_ = opt;
}

int64_t GroupMethod::get_max_cluster_size(int64_t layer_num) {
  return std::max((int64_t)(layer_num / MAX_GROUP_CLUSTER), (int64_t)1);
}

int64_t GroupMethod::cost_add(int64_t cost0, int64_t cost1) {
  if (cost0 == MAX_COST || cost1 == MAX_COST) {
    return MAX_COST;
  } else {
    return (cost0 + cost1);
  }
}

bool GroupMethod::group_one_layer_proc(const LgInfo &lg_info, bool calc_cost,
                                       int64_t *group_cost) {
  if (lg_info.group_ops.size() == 1) {
    if (calc_cost) {
      *group_cost =
          cycle_calculator_->getGlobalLayerCycle(lg_info.group_ops.back());
    }
    return true;
  }
  return false;
}

bool GroupMethod::isLgSupport(Operation *op) {
  bool res = false;
  if (isa<top::WeightOp>(op)) {
    res = true;
  }
  if (auto lg_op = dyn_cast<tpu_mlir::LocalGenInterface>(op)) {
    res = mlir::succeeded(lg_op.LocalGenSupport());
  }
  return res;
}

void GroupMethod::get_base_groups(
    std::vector<std::vector<Operation *>> &base_groups,
    const SetVector<Operation *> &subnet_ops) {
  std::vector<Operation *> group;
  for (auto op : subnet_ops) {
    if (isLgSupport(op)) {
      group.push_back(op);
    } else {
      if (!group.empty()) {
        base_groups.push_back(group);
        group.clear();
      }
      group.push_back(op);
      base_groups.push_back(group);
      group.clear();
    }
  }

  if (!group.empty()) {
    base_groups.push_back(group);
  }
}

static bool group_type_check(const LgInfo &lg_info) {
  auto group_type = lg_info.type;
  for (auto op : lg_info.group_ops) {
    if (isa<MatMulOp>(op)) {
      auto ins = op->getOperands();
      auto Lshape = module::getShape(ins[0]);
      int left_num_dims = Lshape.size();
      int right_num_dims = module::getShape(ins[1]).size();
      if (((left_num_dims == 4 && Lshape[1] < Lshape[2]) ||
           (left_num_dims == 5 && Lshape[1] < Lshape[3])) &&
          right_num_dims == 2) {
        if (group_type != GROUP_SMALL_C) {
          return false;
        }
      }
    }
  }
  return true;
}

static bool group_cslice_check(const LgInfo &lg_info) {
  if (module::isBM1684Family()){
    for (auto op : lg_info.group_ops) {
      if (isa<ActiveOp>(op)) {
        auto shape = module::getShape(op->getOperand(0));
        if (shape.size() > 2 && shape[1] > 4096) {
          return false;
        }
      }
    }
  }
  return true;
}

bool GroupMethod::dynamic_group_valid_check(const LgInfo &lg_info) {
  auto res = true;
  if (runmode_ == RunMode::TPU_DYNAMIC && lg_info.group_ops.size() > 1) {
    // Condition 1
    // Dynamic Backend will choose the first op's batch as the whole group's batch
    // Need make sure dynamic group's ops have the same batch
    int64_t group_n =
        module::getShape(get_output_values(lg_info.group_ops[0])[0])[0];
    for (auto op : lg_info.group_ops) {
      if (!res) break;
      auto outs = get_output_values(op);
      for (auto out : outs) {
        if (group_n != module::getShape(out)[0]) {
          res = false;
          break;
        }
      }
    }
    // Condition 2
    // Inputs and outputs number of a group cannot be large,
    // because it will cause much time to get info of inputs and outputs
    // when dynamic runtime. Also the MCU memory will not be enough
    // to store in/out node.
    const uint32_t max_io_num = 96;
    if (lg_info.group_ins.size() > max_io_num ||
        lg_info.group_outs.size() > max_io_num) {
      res = false;
    }
  }
  return res;
}

bool GroupMethod::group_valid_pre_check(const LgInfo &lg_info) {
  if (!group_type_check(lg_info)) {
    return false;
  }
  if (!group_cslice_check(lg_info)) {
    return false;
  }
  return true;
}

bool GroupMethod::is_layer_group_valid(LgInfo &lg_info, bool calc_cost,
                                       int64_t *group_cost) {
  bool status;
  status = group_one_layer_proc(lg_info, calc_cost, group_cost);
  if (status) {
    return true;
  }

  if (!group_valid_pre_check(lg_info)) {
    return false;
  }

  shape_secs_t shape_secs;
  if(!init_group_data_secs(lg_info, shape_secs)) {
    return false;
  }

  if (!dynamic_group_valid_check(lg_info)) {
    return false;
  }

  auto time_step = std::make_shared<BasicTimeStep>();
  status = time_step->assignTimeStep(lg_info, shape_secs, true);
  if (status == false) {
    return false;
  }

  auto lmem_allocator = std::make_shared<LmemAllocator>();
  status =
      lmem_allocator->assignLmemAddrWithSecs(lg_info, time_step, shape_secs);
  if (status == false) {
    return false;
  }

  if (calc_cost) {
// remove it after pid_node is extractedb
#pragma omp critical(get_cycle)
    *group_cost =
        cycle_calculator_->getGroupCycle(time_step, shape_secs, lg_info.type);
  }
  // llvm::errs() << "nsecs = " << shape_secs.nsecs
  //              << ", hsecs = " << shape_secs.hsecs << "\n";
  return status;
}

void GroupMethod::get_layer_cut_result(
    std::vector<int64_t> &cut_result,
    const std::vector<std::pair<int64_t, int64_t>> &clusters,
    const std::vector<std::vector<int64_t>> &cut_points, int64_t start,
    int64_t end) {
  int64_t opt_cut = cut_points[start][end];
  if (opt_cut != end) {
    get_layer_cut_result(cut_result, clusters, cut_points, start, opt_cut);
    get_layer_cut_result(cut_result, clusters, cut_points, opt_cut + 1, end);
  } else {
    cut_result.push_back(clusters[end].first + clusters[end].second - 1);
  }
}

void GroupMethod::get_group_clusters(
    std::vector<std::pair<int64_t, int64_t>> &clusters,
    const std::vector<Operation *> &base_group) {
  LgInfo sub_group;
  size_t group_layer_num = base_group.size();
  const int64_t max_cluster_size = get_max_cluster_size(group_layer_num);
  int64_t start_idx = 0, end_idx = 1, cluster_size = 1;
  if (max_cluster_size > 1) {
    int64_t pre_cost = 0;
    for (size_t idx = 1; idx < group_layer_num; ++idx) {
      if (start_idx == end_idx - 1) {
        pre_cost =
            cycle_calculator_->getGlobalLayerCycle(base_group[start_idx]);
      }
      pre_cost += cycle_calculator_->getGlobalLayerCycle(base_group[end_idx]);

      int64_t temp_cost = 0;
      get_layer_group(sub_group, base_group, start_idx, end_idx);
      bool is_valid = is_layer_group_valid(sub_group, true, &temp_cost);
      if (is_valid) {
        if (pre_cost <= temp_cost) {
          is_valid = false;
        } else {
          pre_cost = temp_cost;
        }
      }

      if (!is_valid || (is_valid && cluster_size >= max_cluster_size - 1) ||
          idx == group_layer_num - 1) {
        if (is_valid) {
          ++cluster_size;
        }
        clusters.push_back(std::make_pair(start_idx, cluster_size));
        start_idx = is_valid ? end_idx + 1 : end_idx;
        idx = is_valid ? idx + 1 : idx;
        end_idx = start_idx + 1;
        cluster_size = 1;
        pre_cost = 0;
        if ((!is_valid && idx == group_layer_num - 1) ||
            start_idx == group_layer_num - 1) {
          clusters.push_back(std::make_pair(start_idx, cluster_size));
          if (start_idx == group_layer_num - 1) {
            break;
          }
        }
      } else {
        ++cluster_size;
        ++end_idx;
      }
    }
  } else {
    for (size_t layer_idx = 0; layer_idx < group_layer_num; ++layer_idx) {
      clusters.push_back(std::make_pair<int64_t, int64_t>(layer_idx, 1));
    }
  }

  llvm::errs() << "clusters idx(size): ";
  for (size_t i = 0; i < clusters.size(); ++i) {
    llvm::errs() << llvm::format("%d(%d), ", clusters[i].first,
                                 clusters[i].second);
  }
  llvm::errs() << "\n";
}

void GroupMethod::sweep_for_min_cost(
    int64_t *group_cost, int64_t *optimal_point, int64_t start, int64_t end,
    const std::vector<std::vector<int64_t>> &cost_table) {
  for (int64_t sweep = start; sweep < end; ++sweep) {
    int64_t temp_cost =
        cost_add(cost_table[start][sweep], cost_table[sweep + 1][end]);
    if (temp_cost < *group_cost) {
      *group_cost = temp_cost;
      *optimal_point = sweep;
    }
  }
}

void GroupMethod::dynamic_programming_layer_group_with_cluster(
    std::vector<LgInfo> &lg_infos, const SetVector<Operation *> &subnet_ops) {
  llvm::errs() << "\n"
               << "=======================================================\n"
               << "***** Dynamic Programming layer group with cluster ****\n"
               << "=======================================================\n";
  LgInfo sub_group;
  std::vector<std::vector<Operation *>> base_groups;
  get_base_groups(base_groups, subnet_ops);
  llvm::errs() << llvm::format("total num of base_group is %d\n",
                               base_groups.size());
  for (size_t i = 0; i < base_groups.size(); ++i) {
    std::vector<std::pair<int64_t, int64_t>> clusters;
    get_group_clusters(clusters, base_groups[i]);
    size_t cluster_num = clusters.size();
    llvm::errs() << llvm::format(
        "process base group %d, layer_num=%d, cluster_num=%d\n", i,
        base_groups[i].size(), cluster_num);
    if (cluster_num > 1) {
      auto cost_table = std::vector<std::vector<int64_t>>(
          cluster_num, std::vector<int64_t>(cluster_num, 0));
      auto cut_points = std::vector<std::vector<int64_t>>(
          cluster_num, std::vector<int64_t>(cluster_num, 0));
      for (size_t j = 0; j < cluster_num; ++j) {
        int64_t start_idx = clusters[j].first;
        int64_t end_idx = start_idx + clusters[j].second - 1;
        get_layer_group(sub_group, base_groups[i], start_idx, end_idx);
        assert(is_layer_group_valid(sub_group, true, &cost_table[j][j]));
        cut_points[j][j] = j;
      }
      llvm::errs() << "Searching best group slices...\n";
      progressbar bar(cluster_num - 1);
      for (size_t len = 2; len <= cluster_num; ++len) {
        bar.update();
        // llvm::errs() << llvm::format("process cluster len = %d\n", len);
        // #pragma omp parallel for private(sub_group)
        for (int64_t start = 0; start <= cluster_num - len; ++start) {
          int64_t end = start + len - 1;
          // llvm::errs() << "start = " << start << ", end = " << end << "\n";
          int64_t start_idx = clusters[start].first;
          int64_t end_idx = clusters[end].first + clusters[end].second - 1;
          get_layer_group(sub_group, base_groups[i], start_idx, end_idx);

          int64_t group_cost = MAX_COST;
          is_layer_group_valid(sub_group, true, &group_cost);

          int64_t optimal_point = end;
          // sweep_for_min_cost(&group_cost, &optimal_point, start, end,
          //                    cost_table);
          for (int64_t sweep = start; sweep < end; ++sweep) {
            int64_t temp_cost =
                cost_add(cost_table[start][sweep], cost_table[sweep + 1][end]);
            if (temp_cost < group_cost) {
              group_cost = temp_cost;
              optimal_point = sweep;
            }
          }
          cost_table[start][end] = group_cost;
          cut_points[start][end] = optimal_point;
        }
      }
      llvm::errs() << "\n";
      std::vector<int64_t> cut_result;
      get_layer_cut_result(cut_result, clusters, cut_points, 0,
                           cluster_num - 1);
      cut_results_.push_back(std::move(cut_result));
    } else {
      cut_results_.push_back(std::vector<int64_t>(1, 0));
    }
  }

  show_cut_results();
  // some post process for cluster
  llvm::errs() << "-------------------------------------------------------\n";
  llvm::errs() << "Consider redundant computation and gdma cost\n";
  llvm::errs() << "-------------------------------------------------------\n";
  consider_redundant_computation_and_gdma_cost(base_groups);
  show_cut_results();

  llvm::errs() << "-------------------------------------------------------\n";
  llvm::errs() << "Merge cut idx to reduce gdma cost\n";
  llvm::errs() << "-------------------------------------------------------\n";
  bool take_effective = merge_cut_idx_to_reduce_gdma_cost(base_groups);
  show_cut_results();

  if (take_effective) {
    llvm::errs() << "-------------------------------------------------------\n";
    llvm::errs() << "Consider redundant computation and gdma cost again\n"
                 << "due to cut idx merged in the previous step\n";
    llvm::errs() << "-------------------------------------------------------\n";
    consider_redundant_computation_and_gdma_cost(base_groups);
    show_cut_results();
  }

  // update lg_infos
  get_final_groups(lg_infos, base_groups);
}

bool GroupMethod::update_sequence_group_cost(LgInfo *left_layer_group,
                                             LgInfo *right_layer_group,
                                             bool *left_first,
                                             SequenceGroupsInfo &opt_seq_info) {
  assert(left_layer_group->group_ops.size() > 0);
  assert(right_layer_group->group_ops.size() > 0);
  LgInfo *groups[2];
  shape_secs_t *p_shape_secs[2];
  if (*left_first) {
    groups[0] = left_layer_group;
    groups[1] = right_layer_group;
    p_shape_secs[0] = &(opt_seq_info.left_shape_secs);
    p_shape_secs[1] = &(opt_seq_info.right_shape_secs);
  } else {
    groups[0] = right_layer_group;
    groups[1] = left_layer_group;
    p_shape_secs[0] = &(opt_seq_info.right_shape_secs);
    p_shape_secs[1] = &(opt_seq_info.left_shape_secs);
  }
  bool valid = true;
  shape_secs_t shape_secs[2];
  BasicTimeStepPtr time_steps[2] = {std::make_shared<BasicTimeStep>(),
                                    std::make_shared<BasicTimeStep>()};
  auto lmem_allocator = std::make_shared<LmemAllocator>();
  int64_t group_costs[2] = {0, 0};
  bool pre_cost_judge = true;
  for (size_t i = 0; i < 2; ++i) {
    if (group_one_layer_proc(*groups[i], true, &group_costs[i])) {
      shape_secs[i].nsecs = 1;
      shape_secs[i].csecs = 1;
      shape_secs[i].hsecs = 1;
      shape_secs[i].dsecs = 1;
      shape_secs[i].wsecs = 1;
      continue;
    }

    if (!init_group_data_secs(*groups[i], shape_secs[i])) {
      valid = false;
      break;
    }
    if (!time_steps[i]->assignTimeStep(*groups[i], shape_secs[i], true)) {
      valid = false;
      break;
    }
    if (!update_data_split(time_steps[i], *groups[i], shape_secs[i])) {
      valid = false;
      break;
    }

    *left_first = !(*left_first);
    if (pre_cost_judge) {
      if (memcmp(&shape_secs[i], p_shape_secs[i], sizeof(shape_secs_t)) != 0) {
        pre_cost_judge = false;
        continue;
      }
      if (!stripe_mine_max_slice(*groups[i], shape_secs[i],
                                 time_steps[i]->get_tensor_infos())) {
        valid = false;
        break;
      }
      group_costs[i] = cycle_calculator_->getGroupCycle(
          time_steps[i], shape_secs[i], groups[i]->type);
    }
  }
  if (!valid) {
    return false;
  }
  int64_t total_cost = group_costs[0] + group_costs[1];
  if (pre_cost_judge) {
    llvm::errs() << "The pre cost of the two group is " << total_cost << "\n";
    if (opt_seq_info.min_cost >= 0 && opt_seq_info.min_cost < total_cost) {
      return false;
    }
  }

  for (size_t i = 0; i < 2; ++i) {
    if (groups[i]->group_ops.size() == 1) {
      continue;
    }
    if (!lmem_allocator->assignLmemAddrWithSecs(*groups[i], time_steps[i],
                                                shape_secs[i])) {
      valid = false;
      break;
    }
    *left_first = !(*left_first);
    group_costs[i] = cycle_calculator_->getGroupCycle(
        time_steps[i], shape_secs[i], groups[i]->type);
  }
  if (!valid) {
    return false;
  }
  total_cost = group_costs[0] + group_costs[1];
  llvm::errs() << "The final cost of the two group is " << total_cost << "\n";
  if (opt_seq_info.min_cost >= 0 && opt_seq_info.min_cost <= total_cost) {
    return false;
  }
  opt_seq_info.min_cost = total_cost;
  memcpy(p_shape_secs[0], &shape_secs[0], sizeof(shape_secs_t));
  memcpy(p_shape_secs[1], &shape_secs[1], sizeof(shape_secs_t));

  return true;
}

bool GroupMethod::consider_redundant_computation_and_gdma_cost(
    const std::vector<std::vector<Operation *>> &base_groups) {

  int64_t left_cut_idx;
  int64_t optimal_cut_idx;
  SequenceGroupsInfo seq_info;
  LgInfo left_sub_group, right_sub_group;

  for (size_t i = 0; i < base_groups.size(); ++i) {
    auto &base_group = base_groups[i];
    auto &cut_result = cut_results_[i];
    size_t cut_num = cut_result.size();
    if (cut_num > 1 && get_max_cluster_size(base_group.size()) > 1) {
      for (int32_t j = cut_num - 2; j >= 0; --j) {
        left_cut_idx = j > 0 ? (cut_result[j - 1] + 1) : (int64_t)0;

        memset(&seq_info, 0, sizeof(SequenceGroupsInfo));
        seq_info.min_cost = -1;
        optimal_cut_idx = cut_result[j];
        cut_result[j] = cut_result[j + 1] - 1;
        bool left_first = true;
        for (; cut_result[j] >= left_cut_idx; cut_result[j]--) {
          get_layer_group(left_sub_group, base_group, left_cut_idx,
                          cut_result[j]);
          get_layer_group(right_sub_group, base_group, cut_result[j] + 1,
                          cut_result[j + 1]);
          bool is_better = update_sequence_group_cost(
              &left_sub_group, &right_sub_group, &left_first, seq_info);
          if (is_better) {
            optimal_cut_idx = cut_result[j];
            llvm::errs() << "//// Group cost " << seq_info.min_cost
                         << ", optimal cut idx " << optimal_cut_idx << "\n";
          }
        }
        cut_result[j] = optimal_cut_idx;
      }
    }
  }
  return true;
}

bool GroupMethod::merge_cut_idx_to_reduce_gdma_cost(
    const std::vector<std::vector<Operation *>> &base_groups) {
  LgInfo sub_group;
  bool lg_valid;
  bool take_effective = false;
  for (size_t i = 0; i < base_groups.size(); ++i) {
    auto &base_group = base_groups[i];
    auto &cut_result = cut_results_[i];
    if (get_max_cluster_size(base_group.size()) > 1) {
      int64_t left_group_cost = 0, right_group_cost = 0;
      int64_t combine_group_cost = 0;
      size_t size_ = cut_result.size();
      for (size_t j = 0; j < size_ - 1;) {
        size_t cut_idx = cut_result[j];
        size_t start_cut_idx = j > 0 ? (cut_result[j - 1] + 1) : 0;
        size_t end_cut_idx = cut_result[j + 1];
        // get left sub_group
        if (left_group_cost == 0) {
          get_layer_group(sub_group, base_group, start_cut_idx, cut_idx);
          lg_valid = is_layer_group_valid(sub_group, true, &left_group_cost);
          assert(lg_valid);
        }
        // get right sub_group
        get_layer_group(sub_group, base_group, cut_idx + 1, end_cut_idx);
        lg_valid = is_layer_group_valid(sub_group, true, &right_group_cost);
        assert(lg_valid);

        // get combine group
        get_layer_group(sub_group, base_group, start_cut_idx, end_cut_idx);
        lg_valid = is_layer_group_valid(sub_group, true, &combine_group_cost);
        if (lg_valid) {
          if (combine_group_cost < left_group_cost + right_group_cost) {
            cut_result.erase(cut_result.begin() + j);
            size_ = cut_result.size();
            take_effective = true;
            left_group_cost = combine_group_cost;
          } else {
            j++;
            left_group_cost = right_group_cost;
          }
        } else {
          j++;
          left_group_cost = right_group_cost;
        }
      }
    }
  }
  return take_effective;
}

void GroupMethod::simple_layer_group(std::vector<LgInfo> &lg_infos,
                                     const SetVector<Operation *> &subnet_ops) {
  llvm::errs() << "\n"
               << "=======================================================\n"
               << "*********** Group layers as many as possible **********\n"
               << "=======================================================\n";

  cut_results_.clear();
  LgInfo sub_group;
  std::vector<std::vector<Operation *>> base_groups;
  get_base_groups(base_groups, subnet_ops);
  for (int64_t i = base_groups.size() - 1; i >= 0; --i) {
    std::vector<int64_t> cut_result;
    if (base_groups[i].size() == 1) {
      cut_result.push_back(0);
      cut_results_.insert(cut_results_.begin(), std::move(cut_result));
      continue;
    }
    int64_t start_idx = 0, end_idx = base_groups[i].size() - 1;
    cut_result.insert(cut_result.begin(), end_idx);
    while (end_idx > start_idx) {
      get_layer_group(sub_group, base_groups[i], start_idx, end_idx);
      bool valid = is_layer_group_valid(sub_group, false, nullptr);
      if (valid) {
        if (start_idx > 0) {
          cut_result.insert(cut_result.begin(), start_idx - 1);
          end_idx = start_idx - 1;
          start_idx = 0;
        } else {
          break;
        }
      } else {
        start_idx++;
        if (start_idx == end_idx) {
          cut_result.insert(cut_result.begin(), start_idx - 1);
          end_idx = start_idx - 1;
          start_idx = 0;
        }
      }
    }
    cut_results_.insert(cut_results_.begin(), std::move(cut_result));
  }
  show_cut_results();
  get_final_groups(lg_infos, base_groups);
}

void GroupMethod::process(std::vector<LgInfo> &lg_infos,
                          const SetVector<Operation *> &subnet_ops) {
  runmode_ = getRunMode(subnet_ops[0]);
  switch (opt_) {
  case 1:
    simple_layer_group(lg_infos, subnet_ops);
    break;
  case 2:
    dynamic_programming_layer_group_with_cluster(lg_infos, subnet_ops);
    break;
  default:
    simple_layer_group(lg_infos, subnet_ops);
    break;
  }
}

void GroupMethod::get_final_groups(
    std::vector<LgInfo> &lg_infos,
    const std::vector<std::vector<Operation *>> &base_groups) {
  int64_t start_idx, end_idx;
  LgInfo lg_info;
  for (size_t i = 0; i < base_groups.size(); ++i) {
    start_idx = 0;
    auto &base_group = base_groups[i];
    auto &cut_result = cut_results_[i];
    for (size_t j = 0; j < cut_result.size(); ++j) {
      end_idx = cut_result[j];
      get_layer_group(lg_info, base_group, start_idx, end_idx);
      lg_infos.push_back(lg_info);
      start_idx = end_idx + 1;
    }
  }
}

void GroupMethod::show_cut_results() {
  LLVM_DEBUG(for (size_t i = 0; i < cut_results_.size(); ++i) {
    auto &cut_result = cut_results_[i];
    llvm::errs() << "base group[" << i << "] cut results: ";
    for (size_t j = 0; j < cut_result.size(); ++j) {
      llvm::errs() << cut_result[j] << ", ";
    }
    llvm::errs() << "\n";
  });
}

/// The pass of layer group searching
class LayerGroupSearchPass : public LgPass {
public:
  LayerGroupSearchPass(const LgOptions &options) { options_ = options; }
  virtual bool run(LgPassIR *pass_ir) override {
    auto group_method = GroupMethod(options_.opt);
    group_method.process(pass_ir->lg_infos, pass_ir->subnet_ops);
    return true;
  }
  virtual std::string name() override { return "LayerGroupSearchPass"; }
  virtual std::string brief() override {
    return "Searching the optimal layer groups";
  }

private:
  LgOptions options_;
};

std::unique_ptr<LgPass> CreateLayerGroupSearchPass(const LgOptions &options) {
  return std::unique_ptr<LgPass>(new LayerGroupSearchPass(options));
}

} // namespace tpu
} // namespace tpu_mlir

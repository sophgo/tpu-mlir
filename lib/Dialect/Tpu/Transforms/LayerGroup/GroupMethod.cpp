//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupMethod.h"
#include "progressbar.hpp"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Backend/BM168x/BackendInterfaces.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/IlpTimeStep.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgPass.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/TimeStepMethod.h"
#include <llvm/Support/Debug.h>
#include <random>

#define DEBUG_TYPE "layer-group"
using namespace tpu_mlir::backend;

namespace tpu_mlir {
namespace tpu {
#define MAX_GROUP_CLUSTER (50)

#define GROUP_CHECK_RETURN(val)                                                \
  {                                                                            \
    if (val) {                                                                 \
      LAYER_GROUP_LOG_DEBUG_BLOCK(                                             \
          { llvm::outs() << "layer group is valid"; });                        \
      return true;                                                             \
    } else {                                                                   \
      LAYER_GROUP_LOG_DEBUG_BLOCK(                                             \
          { llvm::outs() << "layer group is invalid"; });                      \
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
             SoftmaxOp, RMSNormOp, ReshapeOp, LutOp>(op)) {
      return false;
    }
    if (isa<ReshapeOp>(op)) {
      auto ishape = module::getShape(op->getOperand(0));
      auto oshape = module::getShape(op->getResult(0));
      if (ishape.size() > 5 || oshape.size() > 5) {
        return false;
      }
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
    } else if (auto op_ = dyn_cast<ReshapeOp>(op)) {
      auto ishape = module::getShape(op_.getInput());
      auto oshape = module::getShape(op_.getOutput());
      if (!(ishape.size() > 2 && oshape.size() > 2 && ishape[0] == oshape[0] &&
            ishape[1] == oshape[1])) {
        return false;
      }
      if ((shape.size() == 4 &&
           shape[0] * shape[1] * shape[2] % Arch::NPU_NUM == 0) ||
          (shape.size() == 5 &&
           shape[0] * shape[1] * shape[2] * shape[3] % Arch::NPU_NUM == 0)) {
        return false;
      }
    }

    if ((shape.size() == 4 &&
         shape[0] * shape[1] * shape[2] % Arch::NPU_NUM == 0) ||
        (shape.size() == 5 &&
         shape[0] * shape[1] * shape[2] * shape[3] % Arch::NPU_NUM == 0)) {
      continue;
    }
    if ((shape.size() == 3 && shape[0] > 4 && shape[1] == 197)) {
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
  if (module::isMARS3())
    return false;
  for (auto op : group_ops) {
    if (!isa<ActiveOp, AddOp, CastOp, LayerNormOp, MulConstOp, MatMulOp, MulOp,
             ReshapeOp, SoftmaxOp, AttentionOp, RMSNormOp, MulShiftOp>(op)) {
      return false;
    }
    auto shape = module::getShape(op->getOperand(0));
    if (auto op_ = dyn_cast<LayerNormOp>(op)) {
      if (op_.getAxis() != shape.size() - 1) {
        return false;
      }
      //    } else if (isa<AddOp, MulOp>(op)) {
      //      auto shapeB = module::getShape(op->getOperand(1));
      //      if (shape != shapeB) {
      //        return false;
      //      }
    } else if (auto op_ = dyn_cast<ReshapeOp>(op)) {
      auto ishape = module::getShape(op_.getInput());
      auto oshape = module::getShape(op_.getOutput());
      if (!(ishape.size() > 2 && oshape.size() > 2 && ishape[0] == oshape[0] &&
            ishape[1] == oshape[1])) {
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

static int64_t pair_key(int64_t start, int64_t end) {
  return start << 32 | end;
}

void GroupMethod::set_layer_group_cache(LgInfo lg_info) {
  // llvm::dbgs() << "set_layer_group_cache"
  //              << "; base_group_idx = " << lg_info.base_group_idx
  //              << "; cache_key = " << lg_info.cache_key
  //              << "; group_cost = " << lg_info.group_cost
  //              << "\n";
  lg_cache_[lg_info.base_group_idx][lg_info.cache_key] = std::make_shared<LgInfo>(lg_info);
}

void GroupMethod::get_layer_group(LgInfo &lg_info,
                            const std::vector<Operation *> &base_group,
                            int64_t left, int64_t right, int64_t base_group_idx) {
  auto key = pair_key(left, right);
  if (lg_cache_.find(base_group_idx) != lg_cache_.end() &&
      lg_cache_[base_group_idx].find(key) != lg_cache_[base_group_idx].end()) {

    auto lg_info_ptr = lg_cache_[base_group_idx][key];
    if (lg_info_ptr->is_valid != NOT_CHECK) {
      lg_info = *(lg_info_ptr);
      DEBUG_WITH_TYPE("lg_cache_info", {
        llvm::dbgs() << "; action = lg_cache_info"
                    << "; key = " << key
                    << "; step = hit"
                    << "; base_group_idx = " << base_group_idx
                    << "; start_idx = " << left << "; end_idx = " << right
                    << "\n";
      });
      return;
    }
    return;
  }
  DEBUG_WITH_TYPE("lg_cache_info", {
    llvm::dbgs() << "; action = lg_cache_info"
                << "; key = " << key
                << "; step = miss"
                << "; base_group_idx = " << base_group_idx
                << "; start_idx = " << left
                << "; end_idx = " << right
                << "\n";
  });
  lg_info.clear();
  for (int idx = left; idx <= right; ++idx) {
    lg_info.group_ops.push_back(base_group[idx]);
  }
  lg_info.update_group_io(opt_);
  set_group_type(lg_info);
  lg_info.base_group_idx = base_group_idx;
  lg_info.cache_key = key;
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

bool is_binary_shape_value(Operation *op) {
  if (isa<tpu::AddOp, tpu::SubOp, tpu::MulOp, tpu::DivOp, tpu::MinOp,
          tpu::MaxOp>(op)) {
    auto l_shape = module::getShape(op->getOperand(0));
    auto r_shape = module::getShape(op->getOperand(1));
    if (l_shape.size() == 5 && l_shape[2] != r_shape[2])
      return true;
    else
      return false;
  } else {
    return false;
  }
}

void tmp_group_into_base(std::vector<std::vector<Operation *>> &base_groups,
                         std::vector<Operation *> &group, Operation *op,
                         bool &is_binary) {
  if (isa<Conv3DOp, Pool3DOp>(op) && is_binary) {
    std::vector<Operation *> tmp_group;
    for (auto tmp_op : group) {
      if (!is_binary_shape_value(tmp_op)) {
        tmp_group.push_back(tmp_op);
      } else {
        if (!tmp_group.empty()) {
          base_groups.push_back(tmp_group);
          tmp_group.clear();
        }
        tmp_group.push_back(tmp_op);
        base_groups.push_back(tmp_group);
        tmp_group.clear();
      }
    }
    group = tmp_group;
    is_binary = false;
  }
}

void GroupMethod::get_base_groups(
    std::vector<std::vector<Operation *>> &base_groups,
    const llvm::SetVector<Operation *> &subnet_ops) {
  std::vector<Operation *> group;
  bool is_binary = false;
  for (auto op : subnet_ops) {
    if (isLgSupport(op)) {
      if (!is_binary)
        is_binary = is_binary_shape_value(op);
      group.push_back(op);
      tmp_group_into_base(base_groups, group, op, is_binary);
    } else {
      if (!group.empty()) {
        base_groups.push_back(group);
        group.clear();
      }
      group.push_back(op);
      base_groups.push_back(group);
      group.clear();
      is_binary = false;
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
  if (module::isBM1684Family()) {
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
    // Dynamic Backend will choose the first op's batch as the whole group's
    // batch Need make sure dynamic group's ops have the same batch
    int64_t group_n =
        module::getShape(get_output_values(lg_info.group_ops[0])[0])[0];
    for (auto op : lg_info.group_ops) {
      if (!res)
        break;
      if (isa<tpu::ReshapeOp>(op)) {
        auto reshape_op = dyn_cast<tpu::ReshapeOp>(op);
        auto shape = module::getI64Array(reshape_op.getShape());
        for (auto s : *shape) {
          if (s < 0) {
            res = false;
            break;
          }
        }
      }
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
  if (lg_info.is_valid != NOT_CHECK) {
    *group_cost = lg_info.group_cost;
    return lg_info.is_valid == VALID;
  }
  PROFILE_LOG("is_layer_group_valid", true);
  bool status;
  status = group_one_layer_proc(lg_info, calc_cost, group_cost);
  if (status && LgPass::OPTIONS.group_by_cores == false) {
    PROFILE_LOG("is_layer_group_valid", false);
    lg_info.is_valid = VALID;
    lg_info.group_cost = MAX_COST;
    set_layer_group_cache(lg_info);
    return true;
  }

  if (!group_valid_pre_check(lg_info)) {
    PROFILE_LOG("is_layer_group_valid", false);
    lg_info.is_valid = NOT_VALID;
    lg_info.group_cost = MAX_COST;
    set_layer_group_cache(lg_info);
    return false;
  }

  shape_secs_t shape_secs;
  std::vector<std::pair<Value, int64_t>> value_size;
  if (!init_group_data_secs(lg_info, shape_secs, value_size)) {
    PROFILE_LOG("is_layer_group_valid", false);
    lg_info.is_valid = NOT_VALID;
    lg_info.group_cost = MAX_COST;
    set_layer_group_cache(lg_info);
    return false;
  }
  DEBUG_WITH_TYPE("shape_secs", {
    llvm::dbgs() << "; action = init_group_data_secs"
                 << "; nsecs = " << shape_secs.nsecs
                 << "; csecs = " << shape_secs.csecs
                 << "; dsecs = " << shape_secs.dsecs
                 << "; hsecs = " << shape_secs.hsecs
                 << "; wsecs = " << shape_secs.wsecs << "\n";
  });
  if (!dynamic_group_valid_check(lg_info)) {
    PROFILE_LOG("is_layer_group_valid", false);
    lg_info.is_valid = NOT_VALID;
    lg_info.group_cost = MAX_COST;
    set_layer_group_cache(lg_info);
    return false;
  }

  auto time_step = std::make_shared<BasicTimeStep>();
  status = time_step->assignTimeStep(lg_info, shape_secs, true);
  if (status == false) {
    PROFILE_LOG("is_layer_group_valid", false);
    lg_info.is_valid = NOT_VALID;
    lg_info.group_cost = MAX_COST;
    set_layer_group_cache(lg_info);
    return false;
  }

  auto lmem_allocator = std::make_shared<LmemAllocator>();
  status =
      lmem_allocator->assignLmemAddrWithSecs(lg_info, time_step, shape_secs);
  if (status == false) {
    PROFILE_LOG("is_layer_group_valid", false);
    lg_info.is_valid = NOT_VALID;
    lg_info.group_cost = MAX_COST;
    set_layer_group_cache(lg_info);
    return false;
  }

  if (calc_cost) {
// remove it after pid_node is extractedb
#pragma omp critical(get_cycle)
    *group_cost =
        cycle_calculator_->getGroupCycle(time_step, shape_secs, lg_info.type);
  }
  // llvm::outs() << "nsecs = " << shape_secs.nsecs
  //              << ", hsecs = " << shape_secs.hsecs << "\n";
  PROFILE_LOG("is_layer_group_valid", false);
  lg_info.is_valid = VALID;
  lg_info.shape_secs = shape_secs;
  lg_info.group_cost = *group_cost;
  set_layer_group_cache(lg_info);
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
    const std::vector<Operation *> &base_group, int group_idx) {
  LgInfo sub_group;
  size_t group_layer_num = base_group.size();
  const int64_t max_cluster_size = get_max_cluster_size(group_layer_num);
  int64_t start_idx = 0, end_idx = 1, cluster_size = 1;

  if (max_cluster_size == 1) {
    for (size_t layer_idx = 0; layer_idx < group_layer_num; ++layer_idx) {
      clusters.push_back(std::make_pair<int64_t, int64_t>(layer_idx, 1));
    }
  } else {
    int64_t pre_cost = 0;
    for (size_t idx = 1; idx < group_layer_num; ++idx) {
      if (start_idx == end_idx - 1) {
        pre_cost =
            cycle_calculator_->getGlobalLayerCycle(base_group[start_idx]);
        DEBUG_WITH_TYPE("lg_cost", {
          llvm::dbgs() << "; action = lg_cost"
                       << "; step = get_group_cluster"
                       << "; start_idx = " << start_idx
                       << "; end_idx = " << start_idx
                       << "; group_idx = " << group_idx
                       << "; group_cost = " << pre_cost << "\n";
        });
      }
      int64_t post_cost = cycle_calculator_->getGlobalLayerCycle(base_group[end_idx]);
      pre_cost = cost_add(pre_cost, post_cost);

      DEBUG_WITH_TYPE("lg_cost", {
        llvm::dbgs() << "; action = lg_cost"
                     << "; step = get_group_cluster"
                     << "; start_idx = " << end_idx
                     << "; end_idx = " << end_idx
                     << "; group_idx = " << group_idx
                     << "; group_cost = " << post_cost << "\n";
      });

      int64_t temp_cost = 0;
      get_layer_group(sub_group, base_group, start_idx, end_idx, group_idx);
      bool is_valid = is_layer_group_valid(sub_group, true, &temp_cost);

      DEBUG_WITH_TYPE("lg_cost", {
        llvm::dbgs() << "; action = lg_cost"
                     << "; step = get_group_cluster"
                     << "; start_idx = " << start_idx << "; end_idx = " << end_idx
                     << "; group_idx = " << group_idx
                     << "; group_cost = " << temp_cost << "\n";
      });

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
  }
  LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::outs() << "clusters idx(size): "; });
  for (size_t i = 0; i < clusters.size(); ++i) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({
      llvm::outs() << llvm::format("%d(%d), ", clusters[i].first,
                                   clusters[i].second);
    });
  }
  LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "\n";});

  DEBUG_WITH_TYPE("cluster_info", {
    for (size_t i = 0; i < clusters.size(); ++i) {
      llvm::dbgs() << "action = cluster_info"
                   << "; cluster_idx = " << i
                   << "; start_idx = " << clusters[i].first
                   << "; cluster_size = " << clusters[i].second
                   << "; end_idx = " << clusters[i].first + clusters[i].second - 1
                   << "\n";
    }
  });
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
    std::vector<LgInfo> &lg_infos,
    const llvm::SetVector<Operation *> &subnet_ops) {
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::outs() << "\n"
                 << "=======================================================\n"
                 << "***** Dynamic Programming layer group with cluster ****\n"
                 << "=======================================================\n";
  });
  // for debug
  // std::vector<Operation *> ops_vector;
  // for (Operation *op : subnet_ops) {
  //       ops_vector.push_back(op);
  // }
  // std::shared_ptr<dot_graph> opt2_dot_graph = std::make_shared<dot_graph>();
  // createSubnetGraph(ops_vector, opt2_dot_graph);
  // for debug
  LgInfo sub_group;
  std::vector<std::vector<Operation *>> base_groups;
  get_base_groups(base_groups, subnet_ops);
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::outs() << llvm::format("total num of base_group is %d\n",
                                 base_groups.size());
  });
  for (size_t i = 0; i < base_groups.size(); ++i) {
    std::vector<std::pair<int64_t, int64_t>> clusters;
    get_group_clusters(clusters, base_groups[i], i);
    size_t cluster_num = clusters.size();
    LAYER_GROUP_LOG_DEBUG_BLOCK({
      llvm::outs() << llvm::format(
          "process base group %d, layer_num=%d, cluster_num=%d\n", i,
          base_groups[i].size(), cluster_num);
    });
    if (cluster_num > 1) {
      auto cost_table = std::vector<std::vector<int64_t>>(
          cluster_num, std::vector<int64_t>(cluster_num, 0));
      auto cut_points = std::vector<std::vector<int64_t>>(
          cluster_num, std::vector<int64_t>(cluster_num, 0));
      for (size_t j = 0; j < cluster_num; ++j) {
        int64_t start_idx = clusters[j].first;
        int64_t end_idx = start_idx + clusters[j].second - 1;
        get_layer_group(sub_group, base_groups[i], start_idx, end_idx, i);

        assert(is_layer_group_valid(sub_group, true, &cost_table[j][j]));

        DEBUG_WITH_TYPE("lg_cost",{
            llvm::errs()
                        << "; action = lg_cost"
                        << "; step = global_layer"
                        << "; start_idx = " << start_idx
                         << "; end_idx = " << end_idx
                         << "; start = " << j
                         << "; end = " << j
                         << "; group_idx = " << i
                         << "; group_cost = " << cost_table[j][j] << "\n";
        });

        cut_points[j][j] = j;
      }

      LAYER_GROUP_LOG_DEBUG_BLOCK(
          { llvm::outs() << "Searching best group slices...\n"; });
      progressbar bar(cluster_num - 1);

      /**
       * you can debug any cluster like calc_cost(16, 17);
       */
      auto calc_cost = [&](int64_t start_idx, int64_t end_idx) {
        get_layer_group(sub_group, base_groups[i], start_idx, end_idx, i);
        int64_t group_cost = MAX_COST;
        is_layer_group_valid(sub_group, true, &group_cost);

        return group_cost;
      };

      for (size_t len = 2; len <= cluster_num; ++len) {
        bar.update();
        // llvm::outs() << llvm::format("process cluster len = %d\n", len);
        // #pragma omp parallel for private(sub_group)
        for (int64_t start = 0; start <= cluster_num - len; ++start) {
          int64_t end = start + len - 1;
          // llvm::outs() << "start = " << start << ", end = " << end << "\n";
          int64_t start_idx = clusters[start].first;
          int64_t end_idx = clusters[end].first + clusters[end].second - 1;

          DEBUG_WITH_TYPE("lg_index", {
            llvm::dbgs() << "; action = lg_index"
                          << "; start = " << start
                          << "; end = " << end
                          << "; start_idx = " << start_idx
                          << "; end_idx = " << end_idx
                          << "; group_idx = " << i << "\n";
          });
          DEBUG_WITH_TYPE("lg_index_info", {
            llvm::dbgs() << "; action = lg_index_info: " << i << "\n";
            sub_group.dump_lginfo();
          });
          int64_t group_cost = calc_cost(start_idx, end_idx);
          int64_t optimal_point = end;
          // sweep_for_min_cost(&group_cost, &optimal_point, start, end,
          //                    cost_table);

          DEBUG_WITH_TYPE("lg_cost", {
            llvm::dbgs() << "; action = lg_cost"
                         << "; step = group_layer"
                         << "; start = " << start
                          << "; end = " << end
                          << "; start_idx = " << start_idx
                          << "; end_idx = " << end_idx
                         << "; group_idx = " << i
                         << "; group_cost = " << group_cost << "\n";
          });
          for (int64_t sweep = start; sweep < end; ++sweep) {
            int64_t temp_cost =
                cost_add(cost_table[start][sweep], cost_table[sweep + 1][end]);
            if (temp_cost < group_cost) {
              group_cost = temp_cost;
              optimal_point = sweep;

              DEBUG_WITH_TYPE("lg_cost", {
                llvm::dbgs() << "; action = lg_cost"
                            << "; step = sweep"
                            << "; start = " << start
                            << "; end = " << end
                            << "; sweep = " << sweep
                            << "; group_idx = " << i
                            << "; group_cost = " << group_cost << "\n";
              });
            }
          }

          DEBUG_WITH_TYPE("lg_cost", {
            llvm::dbgs() << "; action = lg_cost"
                         << "; step = update_better"
                         << "; start = " << start
                          << "; end = " << end
                          << "; start_idx = " << start_idx
                          << "; end_idx = " << end_idx
                         << "; group_idx = " << i
                         << "; group_cost = " << group_cost << "\n";
          });

          cost_table[start][end] = group_cost;
          cut_points[start][end] = optimal_point;

          DEBUG_WITH_TYPE("cut_points", {
            llvm::dbgs() << "; action = cut_points"
                         << "; start = " << start
                         << "; end = " << end
                         << "; cut_points = " << cut_points[start][end]
                         << "; start_idx = " << start_idx
                         << "; end_idx = " << end_idx
                         << "; group_idx = " << i
                         << "\n";
          });
        }
      }
      llvm::outs() << "\n";
      std::vector<int64_t> cut_result;
      get_layer_cut_result(cut_result, clusters, cut_points, 0,
                           cluster_num - 1);
      cut_results_.push_back(std::move(cut_result));
      LLVM_DEBUG({
        LgInfo sub_group;
        int start = 0;
        for (auto end : cut_result) {
          get_layer_group(sub_group, base_groups[i], start, end, i);
          int64_t group_cost = MAX_COST;
          auto temp_status = is_layer_group_valid(sub_group, true, &group_cost);
          llvm::dbgs() << temp_status << " ;start" << start << " - "
                       << " end " << end << " = " << group_cost << "\n";
          start = end + 1;
        }

        llvm::dbgs() << "\n";
        llvm::dbgs() << "================FINAL GROUP================\n";
        for (size_t cost_i = 0; cost_i < cluster_num; ++cost_i) {
          for (int64_t cost_j = 0; cost_j < cluster_num; ++cost_j) {
            llvm::dbgs() << cut_points[cost_i][cost_j] << ", "
                         << "";
          }
          llvm::dbgs() << "\n";
        }
        llvm::dbgs() << "================COST TABLE================\n";
        for (size_t cost_i = 0; cost_i < cluster_num; ++cost_i) {
          for (int64_t cost_j = 0; cost_j < cluster_num; ++cost_j) {
            llvm::dbgs() << cost_table[cost_i][cost_j] << ", "
                         << "";
          }
          llvm::dbgs() << "\n";
        }
        llvm::dbgs() << "=============================================\n";
        llvm::dbgs() << "\n";
      });
    } else {
      cut_results_.push_back(std::vector<int64_t>(1, 0));
      DEBUG_WITH_TYPE("lg_cost", {
        if (!isa<ReturnOp>(base_groups[i][0]) &&
            runmode_ == RunMode::TPU_STATIC) {
          int64_t start_idx = clusters[0].first;
          // int64_t end_idx = start_idx + clusters[0].second - 1;
          get_layer_group(sub_group, base_groups[i], start_idx, start_idx, i);
          int64_t cost;

          assert(is_layer_group_valid(sub_group, true, &cost));
          llvm::dbgs()  << "; action = lg_cost"
                        << "; step = global_layer"
                        << "; start_idx = " << 0 << "; end_idx = " << 0
                        << "; start = " << 0 << "; end = " << 0
                        << "; group_idx = " << i << "; group_cost = " << cost
                       << "\n";
        } else {
          llvm::dbgs() << "; action = lg_cost"
                       << "; step = global_layer"
                       << "; start_idx = " << 0 << "; end_idx = " << 0
                       << "; start = " << 0 << "; end = " << 0
                       << "; group_idx = " << i << "; group_cost = " << 0
                       << "\n";
        }
      });
    }
  }

  show_cut_results();
  // some post process for cluster
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::outs() << "-------------------------------------------------------\n";
    llvm::outs() << "Consider redundant computation and gdma cost\n";
    llvm::outs() << "-------------------------------------------------------\n";
  });
  consider_redundant_computation_and_gdma_cost(base_groups);
  show_cut_results();

  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::outs() << "-------------------------------------------------------\n";
    llvm::outs() << "Merge cut idx to reduce gdma cost\n";
    llvm::outs() << "-------------------------------------------------------\n";
  });
  bool take_effective = merge_cut_idx_to_reduce_gdma_cost(base_groups);
  show_cut_results();

  if (take_effective) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({
      llvm::outs()
          << "-------------------------------------------------------\n";
      llvm::outs() << "Consider redundant computation and gdma cost again\n"
                   << "due to cut idx merged in the previous step\n";
      llvm::outs()
          << "-------------------------------------------------------\n";
    });
    consider_redundant_computation_and_gdma_cost(base_groups);
    show_cut_results();
  }

  // for debug, fix cut results
  // std::vector<int64_t> override_is = {8, 10, 12, 16, 24, 26, 36, 42, 77, 91,
  // 126, 133}; std::vector<int64_t> override_is = {8, 10, 12, 20, 22, 24, 26,
  // 32, 34, 36, 44, 45, 46, 47, 54, 58, 74, 76, 78, 90, 98, 105, 107, 109, 112,
  // 119, 120, 121, 122, 126, 133}; cut_results_[0] = override_is;
  // show_cut_results();

  // update lg_infos
  get_final_groups(lg_infos, base_groups);

  // for debug
  // int grp_idx = 0;
  // for (auto lg_info : lg_infos) {
  //   if(lg_info.group_ops.size()>1){
  //     for (auto op : lg_info.group_ops) {
  //       if(!isa<ReturnOp>(op)){
  //         auto name = module::getName(op).str();
  //         opt2_dot_graph->add_node_label(name + "_ori",
  //                                       "grp_" + std::to_string(grp_idx));
  //       }
  //     }
  //     grp_idx++;
  //   }
  // }
  // std::cout<<"attention !!! opt2 grp"<<grp_idx<<std::endl;
  // opt2_dot_graph->export_dot("opt2_ok");
  // for debug
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
  // bool pre_cost_judge = true;

  for (size_t i = 0; i < 2; ++i) {
    auto status = is_layer_group_valid(*groups[i], true, &group_costs[i]);
    if (status == false) {
      valid = false;
      break;
    }
  }
  if (!valid) {
    return false;
  }
  int64_t total_cost = group_costs[0] + group_costs[1];
  LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "The final cost of the two group is " << total_cost << "\n";});
  if (opt_seq_info.min_cost >= 0 && opt_seq_info.min_cost <= total_cost) {
    return false;
  }
  opt_seq_info.min_cost = total_cost;
  opt_seq_info.left_cost = group_costs[0];
  opt_seq_info.right_cost = group_costs[1];
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

        DEBUG_WITH_TYPE("lg_index", {
            llvm::dbgs() << "; action = lg_index"
                         << "; step = consider_redundant_computation_and_gdma_cost"
                         << "; start_idx = " << left_cut_idx
                         << "; end_idx = " << cut_result[j]
                         << "; group_idx = " << i
                         << "\n";
        });

        memset(&seq_info, 0, sizeof(SequenceGroupsInfo));
        seq_info.min_cost = -1;
        optimal_cut_idx = cut_result[j];
        cut_result[j] = cut_result[j + 1] - 1;
        bool left_first = true;
        for (; cut_result[j] >= left_cut_idx; cut_result[j]--) {
          get_layer_group(left_sub_group, base_group, left_cut_idx,
                          cut_result[j], i);
          get_layer_group(right_sub_group, base_group, cut_result[j] + 1,
                          cut_result[j + 1], i);
          bool is_better = update_sequence_group_cost(
              &left_sub_group, &right_sub_group, &left_first, seq_info);

          DEBUG_WITH_TYPE("lg_cost", {
            llvm::dbgs() << "; action = " << "lg_cost"
                         << "; start_idx = " << left_cut_idx
                         << "; end_idx = " << cut_result[j]
                         << "; group_cost = " << seq_info.left_cost
                         << "; group_idx = " << i << "; step = "
                         << "consider_redundant_computation_and_gdma_cost"
                         << "; part = " << "left"
                         << "\n";
            llvm::dbgs() << "; action = " << "lg_cost"
                         << "; start_idx = " << cut_result[j] + 1
                         << "; end_idx = " << cut_result[j + 1]
                         << "; group_cost = " << seq_info.right_cost
                         << "; group_idx = " << i << "; step = "
                         << "consider_redundant_computation_and_gdma_cost"
                         << "; part = " << "right"
                         << "\n";
          });
          if (is_better) {
            optimal_cut_idx = cut_result[j];
            LAYER_GROUP_LOG_DEBUG_BLOCK({
              llvm::outs() << "//// Group cost " << seq_info.min_cost
                           << ", optimal cut idx " << optimal_cut_idx << "\n";
            });
          }
        }
        cut_result[j] = optimal_cut_idx;

        DEBUG_WITH_TYPE("cut_optimize", {
          llvm::dbgs() << "; action = cut_optimize"
                       << "; step = consider_redundant_computation_and_gdma_cost"
                       << "; left_range = " << left_cut_idx << "-" << optimal_cut_idx
                       << "; right_range = " << optimal_cut_idx + 1 << "-" << cut_result[j + 1]
                       << "; group_idx = " << i << "\n";
        });
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
          get_layer_group(sub_group, base_group, start_cut_idx, cut_idx, i);
          lg_valid = is_layer_group_valid(sub_group, true, &left_group_cost);
          assert(lg_valid);
          DEBUG_WITH_TYPE("lg_cost", {
            llvm::dbgs() << "; start_idx = " << start_cut_idx
                         << "; end_idx = " << cut_idx
                         << "; group_cost = " << left_group_cost
                         << "; group_idx = " << i
                         << "; action = " << "lg_cost"
                         << "; step = " << "merge_cut_idx_to_reduce_gdma_cost"
                         << "; part = " << "left"
                         << "\n";
          });
        }
        // get right sub_group
        get_layer_group(sub_group, base_group, cut_idx + 1, end_cut_idx, i);
        lg_valid = is_layer_group_valid(sub_group, true, &right_group_cost);
        assert(lg_valid);

        DEBUG_WITH_TYPE("lg_cost", {
          llvm::dbgs() << "; start_idx = " << cut_idx + 1
                       << "; end_idx = " << end_cut_idx
                       << "; group_cost = " << right_group_cost
                       << "; group_idx = " << i
                       << "; action = " << "lg_cost"
                       << "; step = " << "merge_cut_idx_to_reduce_gdma_cost"
                       << "; part = " << "right"
                       << "\n";
        });

        // get combine group
        get_layer_group(sub_group, base_group, start_cut_idx, end_cut_idx, i);
        lg_valid = is_layer_group_valid(sub_group, true, &combine_group_cost);
        if (lg_valid) {
          if (combine_group_cost < left_group_cost + right_group_cost) {
            DEBUG_WITH_TYPE("lg_cost", {
              llvm::dbgs() << "; start_idx = " << start_cut_idx
                           << "; end_idx = " << end_cut_idx
                           << "; group_cost = " << combine_group_cost
                           << "; group_idx = " << i
                           << "; action = " << "lg_cost"
                           << "; step = " << "merge_cut_idx_to_reduce_gdma_cost"
                           << "\n";
            });
            DEBUG_WITH_TYPE("cut_optimize", {
              llvm::dbgs() << "; action = cut_optimize"
                           << "; step = merge_cut_idx"
                           << "; left_range = " << start_cut_idx << "-" << cut_idx
                           << "; right_range = " << cut_idx + 1 << "-" << end_cut_idx
                           << "; group_idx = " << i << "\n";
            });
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

void GroupMethod::simple_layer_group(
    std::vector<LgInfo> &lg_infos,
    const llvm::SetVector<Operation *> &subnet_ops) {
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::outs() << "\n"
                 << "=======================================================\n"
                 << "*********** Group layers as many as possible **********\n"
                 << "=======================================================\n";
  });
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
      get_layer_group(sub_group, base_groups[i], start_idx, end_idx, i);
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

void GroupMethod::process(LgPassIR *pass_ir) {
  std::vector<LgInfo> &lg_infos = pass_ir->lg_infos;
  llvm::SetVector<Operation *> &subnet_ops = pass_ir->subnet_ops;
  auto start = std::chrono::high_resolution_clock::now();
  runmode_ = getRunMode(subnet_ops[0]);
  // auto func_name = pass_ir->func.getName();

  if (getenv("LOAD_TPU_GROUP") || LgPass::OPTIONS.opt == 4) {
    if (is_lg_results_exists()) {
      load_lg_results(lg_infos, subnet_ops);
    } else {
      llvm_unreachable("file not exist's, ues opt=1/2/3 to generate");
    }
  } else {
    switch (LgPass::OPTIONS.opt) {
    case 1:
      simple_layer_group(lg_infos, subnet_ops);
      dump_lg_results(lg_infos);
      break;
    case 2:
      dynamic_programming_layer_group_with_cluster(lg_infos, subnet_ops);
      dump_lg_results(lg_infos);
      break;
    case 3:
      ilp_layer_group(pass_ir);
      dump_lg_results(lg_infos);
      break;
      break;
    default:
      simple_layer_group(lg_infos, subnet_ops);
      break;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  llvm::errs() << "GroupMethod_process time:" << elapsed.count() << "\n";
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
      get_layer_group(lg_info, base_group, start_idx, end_idx, i);
      int64_t cost = -1;

      if (base_group.size() > 1) {
        if (!is_layer_group_valid(lg_info, true, &cost)) {
          llvm_unreachable("group_cost is not valid");
        }
      }
      if (lg_info.group_ops.size() > 1 ||
          false == LgPass::OPTIONS.group_by_cores) {
        lg_infos.push_back(lg_info);
      }
      DEBUG_WITH_TYPE("lg_results", {
        if (runmode_ == RunMode::TPU_STATIC) {
          llvm::dbgs() << "; action = lg_results"
                       << "; start_idx = " << start_idx
                       << "; end_idx = " << end_idx << "; group_cost = " << cost
                       << "; final_group_idx = " << i << "\n";
          lg_info.dump_lginfo();
        }
      });
      start_idx = end_idx + 1;
    }
  }
}

void GroupMethod::show_cut_results() {
  DEBUG_WITH_TYPE("lg_results", {
    for (size_t i = 0; i < cut_results_.size(); ++i) {
      auto &cut_result = cut_results_[i];
      llvm::dbgs() << "base group[" << i << "] cut results: ";
      for (size_t j = 0; j < cut_result.size(); ++j) {
        llvm::dbgs() << cut_result[j] << ", ";
      }
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "\n";
  });

  DEBUG_WITH_TYPE("cut_optimize", {
    for (size_t i = 0; i < cut_results_.size(); ++i) {
      auto &cut_result = cut_results_[i];
      int64_t start_idx = 0;
      for (size_t j = 0; j < cut_result.size() - 1; ++j) {
        int64_t end_idx = cut_result[j];
        llvm::dbgs() << "; action = cut_optimize"
                     << "; step = show_cut_results"
                     << "; range = " << start_idx << "-" << end_idx
                     << "; group_idx = " << i << "\n";
        start_idx = end_idx + 1;
      }
    }
  });
}

bool GroupMethod::is_lg_results_exists() {
  auto filename = "layer_group_cache." +
                  module::getName(module::getModuleOp()).str() + ".json";
  auto ret = std::filesystem::exists(filename);
  if (!ret) {
    llvm::errs() << filename << "not exists\n";
  }
  return ret;
}
void GroupMethod::dump_lg_results(std::vector<LgInfo> &lg_infos) {
  if (!LgPass::OPTIONS.lgcache) {
    return;
  }

  std::error_code EC;
  llvm::raw_fd_ostream OS("layer_group_cache." +
                              module::getName(module::getModuleOp()).str() +
                              ".json",
                          EC);
  if (EC) {
    llvm::errs() << "Failed to open file for writing: " << EC.message() << "\n";
    return;
  }

  json::OStream J(OS, 2);
  J.objectBegin();

  J.attribute("opt", LgPass::OPTIONS.opt);
  // Write GroupLayer array
  J.attributeBegin("GroupLayer");
  J.arrayBegin();

  for (const auto &it : llvm::enumerate(lg_infos)) {
    auto group = it.value();
    if (group.group_ops.size() <= 1) {
      continue;
    }
    auto index = it.index();
    llvm::dbgs() << index << "\n";

    J.objectBegin();
    J.attribute("index", index);
    J.attribute("group_cost", group.group_cost);
    // Write locs array
    J.attributeArray("locs", [&] {
      for (const auto &loc : group.group_ops) {
        if (loc) {
          J.value(module::getName(loc));
        }
      }
    });

    // Write shape_secs if available
    J.attributeArray("shape_secs", [&] {
      J.value(group.shape_secs.nsecs);
      J.value(group.shape_secs.csecs);
      J.value(group.shape_secs.dsecs);
      J.value(group.shape_secs.hsecs);
      J.value(group.shape_secs.wsecs);
    });

    J.objectEnd();
  }
  J.arrayEnd();
  J.attributeEnd();

  J.attributeArray("GlobalLayer", [&] {
    for (const auto &it : llvm::enumerate(lg_infos)) {
      auto index = it.index();
      auto layer = it.value();
      if (layer.group_ops.size() > 1) {
        continue;
      }
      J.objectBegin();
      J.attribute("index", index);
      J.attribute("group_cost", layer.group_cost);
      J.attribute("loc", module::getName(layer.group_ops[0]).str());
      J.objectEnd();
    }
  });

  J.objectEnd();
}

void GroupMethod::load_lg_results(
    std::vector<LgInfo> &lg_infos,
    const llvm::SetVector<Operation *> &subnet_ops) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(
      "layer_group_cache." + module::getName(module::getModuleOp()).str() +
      ".json");
  if (!bufferOrErr) {
    llvm::errs() << "Failed to open file: " << bufferOrErr.getError().message()
                 << "\n";
    return;
  }

  lg_infos.clear();

  std::map<std::string, Operation *> op_map;
  for (auto op : subnet_ops) {
    op_map[module::getName(op).str()] = op;
  }
  // Parse JSON
  auto jsonOrErr = json::parse((*bufferOrErr)->getBuffer());
  if (!jsonOrErr) {
    llvm::errs() << "Failed to parse JSON: " << toString(jsonOrErr.takeError())
                 << "\n";
    return;
  }

  auto &root = *jsonOrErr;
  int opt = LgPass::OPTIONS.opt;
  // Load group layers
  if (auto *rootObj = root.getAsObject()) {
    if (auto opt_ = rootObj->getInteger("opt")) {
      opt = *opt_;
    } else {
      llvm_unreachable("opt not found");
    }

    if (auto groupLayerArray = rootObj->getArray("GroupLayer")) {
      for (const auto &groupObj : *groupLayerArray) {
        LgInfo lg_info;

        if (auto *groupObj_ = groupObj.getAsObject()) {
          // Get operation locations
          if (auto index = groupObj_->getInteger("index")) {
            lg_info.group_id = *index;
          }
          if (auto group_cost = groupObj_->getInteger("group_cost")) {
            lg_info.group_cost = *group_cost;
          }
          if (auto locsArray = groupObj_->getArray("locs")) {
            for (const auto &loc : *locsArray) {
              if (auto opName = loc.getAsString()) {
                // Find operation by name and add to group
                if (auto op = op_map[opName->str()]) {
                  lg_info.group_ops.push_back(op);
                }
              }
            }
          }
          // Get shape_secs if available
          if (auto shapeArray = groupObj_->getArray("shape_secs")) {
            std::vector<int64_t> shape_values;
            for (const auto &val : *shapeArray) {
              if (auto num = val.getAsInteger()) {
                shape_values.push_back(*num);
              }
            }
            if (shape_values.size() == 5 && shape_values[0] != 0) {
              lg_info.shape_secs.nsecs = shape_values[0];
              lg_info.shape_secs.csecs = shape_values[1];
              lg_info.shape_secs.dsecs = shape_values[2];
              lg_info.shape_secs.hsecs = shape_values[3];
              lg_info.shape_secs.wsecs = shape_values[4];
            } else {
              // to indicate running shape_secs search in assignLmemAddrWithSecs
              lg_info.shape_secs.nsecs = 0;
            }
          }
        }

        // Add group if valid
        if (!lg_info.group_ops.empty()) {
          lg_info.update_group_io(opt);
          set_group_type(lg_info);
          lg_infos.push_back(lg_info);
        }
      }
    }

    // Load global layers
    if (auto globalArray = rootObj->getArray("GlobalLayer")) {
      for (const auto &globalObj : *globalArray) {
        LgInfo lg_info;
        if (auto *globalObj_ = globalObj.getAsObject()) {
          if (auto index = globalObj_->getInteger("index")) {
            lg_info.group_id = *index;
          }
          if (auto group_cost = globalObj_->getInteger("group_cost")) {
            lg_info.group_cost = *group_cost;
          }
          if (auto opName = globalObj_->getString("loc")) {
            if (auto op = op_map[opName->str()]) {
              lg_info.group_ops.push_back(op);
              lg_info.update_group_io(opt);
              set_group_type(lg_info);
              lg_infos.push_back(lg_info);
            }
          }
        }
      }
    }
  }

  // Sort lg_infos by index if needed
  std::sort(
      lg_infos.begin(), lg_infos.end(),
      [](const LgInfo &a, const LgInfo &b) { return a.group_id < b.group_id; });

  for (auto &lg_info : lg_infos) {
    int64_t cost = 0;
    lg_info.use_cache = true;
    if (lg_info.group_cost > 0) {
      if (!is_layer_group_valid(lg_info, true, &cost)) {
        llvm_unreachable("group_cost is not valid");
      }
    }
  }
}

/// The pass of layer group searching
class LayerGroupSearchPass : public LgPass {
public:
  LayerGroupSearchPass(const LgOptions &options) { options_ = options; }
  virtual bool run(LgPassIR *pass_ir) override {
    auto group_method = GroupMethod(options_.opt);
    group_method.process(pass_ir);
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

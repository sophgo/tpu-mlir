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
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/Debugger.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/IlpTimeStep.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgCache.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgConfig.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgPass.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/TimeStepMethod.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/WithColor.h>
#include <llvm/Support/raw_ostream.h>
#include <random>

#define DEBUG_TYPE "layer-group"
static int subfunc_idx = 0;
#define CACHE_FILE_NAME                                                        \
  module::getName(module::getModuleOp()).str() + "_" +                         \
      module::getChipStr().str() + "_" + module::getModeStr() +                \
      ".layer_group_cache.json"
#define DEBUGGER_FILE_NAME                                                     \
  module::getName(module::getModuleOp()).str() + "_" +                         \
      module::getChipStr().str() + "_" + module::getModeStr() +                \
      ".layer_group_debugger.json"
#define IDX_FILE_NAME                                                          \
  module::getName(module::getModuleOp()).str() + "_" +                         \
      module::getChipStr().str() + "_" + module::getModeStr() +                \
      ".layer_group_idx.txt"
#define HASH_FILE_NAME std::to_string(modules_hash)
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
    if (auto layernorm_op = dyn_cast<LayerNormOp>(op)) {
      if (module::getShape(layernorm_op.getInput()).size() == 5) {
        return true;
      }
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
    if (!module::isTpuArOp(op) &&
        !isa<ActiveOp, CastOp, LayerNormOp, MulConstOp, MatMulOp, SoftmaxOp,
             RMSNormOp, ReshapeOp, LutOp, BinaryShiftOp>(op)) {
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
    } else if (module::isTpuArOp(op)) {
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
      if (ishape.size() == 4 && oshape.size() > 2 && ishape[2] != oshape[2]) {
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
  if (module::isCV184X() || module::isSGTPUV8()) {
    /** NOTE: following condition may need to  be removed.*/
    return false;
  }
  for (auto op : group_ops) {
    if (!isa<ActiveOp, AddOp, CastOp, LayerNormOp, MulConstOp, MatMulOp, MulOp,
             ReshapeOp, SoftmaxOp, AttentionOp, RMSNormOp, MulShiftOp, WhereOp,
             BatchNormBwdOp, LutOp, BinaryConstShiftOp, BinaryShiftOp,
             AddConstOp, ReduceOp, ClipOp, DivOp, RopeOp, ConcatOp>(op)) {
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
  // lg_cache_[lg_info.base_group_idx][lg_info.cache_key] =
  //     std::make_shared<LgInfo>(lg_info);
}

bool GroupMethod::get_layer_group(LgInfo &lg_info,
                                  const std::vector<Operation *> &base_group,
                                  int64_t left, int64_t right,
                                  int64_t base_group_idx, int64_t idx_offset) {
  auto key = pair_key(left, right);
  // if (lg_cache_.find(base_group_idx) != lg_cache_.end() &&
  //     lg_cache_[base_group_idx].find(key) != lg_cache_[base_group_idx].end())
  //     {

  //   auto lg_info_ptr = lg_cache_[base_group_idx][key];
  //   if (lg_info_ptr->is_valid != NOT_CHECK) {
  //     lg_info = *(lg_info_ptr);
  //     DEBUG_WITH_TYPE("lg_cache_info", {
  //       llvm::dbgs() << "; action = lg_cache_info"
  //                    << "; key = " << key << "; step = hit"
  //                    << "; base_group_idx = " << base_group_idx
  //                    << "; start_idx = " << left << "; end_idx = " << right
  //                    << "\n";
  //     });
  //     return;
  //   }
  // }
  lg_info.clear();
  LG_DEBUG_WITH_TYPE("lg_step", [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "set_lg_info", "stamp",
                        "setting: group info(group_ops, group_ins, group_outs, "
                        "group_op_outs, type) & "
                        "cache info(base_group_idx, start_idx, end_idx, "
                        "func_start_idx, func_end_idx, cache_key)")
                 << "\n";
  });
  auto base_group_size = (int64_t)base_group.size();
  if (left > right || left < 0 || right < 0 || left >= base_group_size ||
      right >= base_group_size) {
    LG_DEBUG_WITH_TYPE("lg_info", [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "get_layer_group", "end",
                          "invalid layer group indices, return false")
                   << "\n";
    });
    return false;
  }
  for (int idx = left; idx <= right; ++idx) {
    lg_info.group_ops.push_back(base_group[idx]);
  }
  lg_info.update_group_io(options_.opt);
  set_group_type(lg_info);
  lg_info.cache_key = key;
  lg_info.base_group_idx = base_group_idx;
  lg_info.start_idx = left;
  lg_info.end_idx = right;
  lg_info.func_start_idx = left + idx_offset;
  lg_info.func_end_idx = right + idx_offset;

  GROUP_DEBUG_WITH_TYPE("lg_info", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO("get_layer_group", "end",
                                          "show current lg_info")
                 << "\n";
    lg_info.dump();
  });
  return true;
}

GroupMethod::GroupMethod(const LgOptions &options) {
  options_ = options;
  MAX_COST = llvm::maxIntN(64);

  if (module::isCV18xx()) {
    Cv18xxCycleCalculator *cyc_ptr =
        new Cv18xxCycleCalculator(options_.num_core);
    cycle_calculator_ = std::shared_ptr<CycleCalculator>(cyc_ptr);
  } else {
    Bm168xCycleCalculator *cyc_ptr =
        new Bm168xCycleCalculator(options_.num_core);
    cycle_calculator_ = std::shared_ptr<CycleCalculator>(cyc_ptr);
  }
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

bool GroupMethod::group_one_layer_proc(LgInfo &lg_info, bool calc_cost,
                                       int64_t *group_cost) {
  if (lg_info.group_ops.size() == 1) {
    if (calc_cost) {
      *group_cost =
          cycle_calculator_->getGlobalLayerCycle(lg_info.group_ops.back())
              .cycles;
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

void GroupMethod::get_debug_group(
    std::vector<Operation *> &debug_group,
    const llvm::SetVector<Operation *> &subnet_ops, int64_t start_idx,
    int64_t end_idx) {
  int idx = 0;
  for (auto op : subnet_ops) {
    if (idx > end_idx) {
      break;
    }
    if (idx >= start_idx) {
      debug_group.push_back(op);
    }
    idx++;
  }
}

void GroupMethod::get_debug_cluster(
    std::vector<std::pair<int64_t, int64_t>> &clusters, int64_t cluster_num) {
  for (auto i = 0; i < cluster_num; i++) {
    clusters.push_back(std::make_pair(i, 1)); // (start_idx=i, cluster_size=1)
  }
}

static bool group_type_check(LgInfo &lg_info) {
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

static bool group_cslice_check(LgInfo &lg_info) {
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

bool GroupMethod::dynamic_group_valid_check(LgInfo &lg_info) {
  auto res = true;
  if (runmode_ == RunMode::TPU_DYNAMIC && lg_info.group_ops.size() > 1) {
    auto disable_dynamic_layer_group =
        std::getenv("DISABLE_DYNAMIC_LAYER_GROUP");
    if (disable_dynamic_layer_group)
      return false;
    // return false;
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

bool GroupMethod::group_valid_pre_check(LgInfo &lg_info) {
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
  // if (lg_info.is_valid != NOT_CHECK) {
  //   *group_cost = lg_info.group_cost;
  //   return lg_info.is_valid == VALID;
  // }
  PROFILE_LOG("is_layer_group_valid", true);
  bool status;
  GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "group_one_layer_proc", "call_function",
                        "if group has only one layer, calculate the cost of "
                        "the layer and return")
                 << "\n";
  });
  status = group_one_layer_proc(lg_info, calc_cost, group_cost);
  // if (status && options_.group_by_cores == false) {
  if (status) {
    PROFILE_LOG("is_layer_group_valid", false);
    lg_info.is_valid = VALID;
    lg_info.group_cost = *group_cost;
    set_layer_group_cache(lg_info);
    GROUP_DEBUG_WITH_TYPE("lg_result", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "group_one_layer_proc", "success",
                          "group with only one layer is always valid")
                   << "\n";
    });
    return true;
  }

  GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO("group_valid_pre_check",
                                          "call_function",
                                          "check group's validity through "
                                          "specific ops' info and group type")
                 << "\n";
  });
  if (!group_valid_pre_check(lg_info)) {
    PROFILE_LOG("is_layer_group_valid", false);
    lg_info.is_valid = NOT_VALID;
    lg_info.group_cost = MAX_COST;
    set_layer_group_cache(lg_info);
    GROUP_DEBUG_WITH_TYPE("lg_result", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "group_valid_pre_check", "failed",
                          "failed to match rules setting for group's `lg_info`")
                   << "\n";
    });
    return false;
  }

  shape_secs_t shape_secs;
  std::vector<std::pair<Value, int64_t>> value_size;
  GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
    llvm::dbgs()
        << DEBUGGER_DEFAULT_INFO(
               "init_group_data_secs", "call_function",
               "get group's `init shape_secs` which is a upper bound of "
               "all valid `shape_secs` according to chip and op features")
        << "\n";
  });
  if (!init_group_data_secs(lg_info, shape_secs, value_size, options_)) {
    PROFILE_LOG("is_layer_group_valid", false);
    lg_info.is_valid = NOT_VALID;
    lg_info.group_cost = MAX_COST;
    set_layer_group_cache(lg_info);
    GROUP_DEBUG_WITH_TYPE("lg_result", lg_info, [&]() {
      llvm::dbgs()
          << DEBUGGER_DEFAULT_INFO(
                 "init_group_data_secs", "failed",
                 "local memory isn't enough to store all tensors even if use "
                 "`init shape_secs` which is a upper bound of all valid "
                 "`shape_secs`")
          << "\n";
    });
    return false;
  }
  GROUP_DEBUG_WITH_TYPE("shape_secs", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "init_group_data_secs_success", "stamp",
                        "find a upper bound of all valid `shape_secs`")
                 << LOG_KV("nsecs", shape_secs.nsecs)
                 << LOG_KV("csecs", shape_secs.csecs)
                 << LOG_KV("dsecs", shape_secs.dsecs)
                 << LOG_KV("hsecs", shape_secs.hsecs)
                 << LOG_KV("wsecs", shape_secs.wsecs) << "\n";
  });

  GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
    llvm::dbgs()
        << DEBUGGER_DEFAULT_INFO(
               "dynamic_group_valid_check", "call_function",
               "check whether the group with dynamic run mode is valid")
        << "\n";
  });
  if (!dynamic_group_valid_check(lg_info)) {
    PROFILE_LOG("is_layer_group_valid", false);
    lg_info.is_valid = NOT_VALID;
    lg_info.group_cost = MAX_COST;
    set_layer_group_cache(lg_info);
    GROUP_DEBUG_WITH_TYPE("lg_result", lg_info, [&]() {
      llvm::dbgs()
          << DEBUGGER_DEFAULT_INFO(
                 "dynamic_group_valid_check", "failed",
                 "failed to match the rules setting for dynamic subnet")
          << "\n";
    });
    return false;
  }

  auto time_step = std::make_shared<BasicTimeStep>(options_);
  GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
    llvm::dbgs()
        << DEBUGGER_DEFAULT_INFO(
               "assignTimeStep", "call_function",
               "backward `slice_info` of tensors starting from output tensors "
               "according to `init shape_secs`; "
               "assign and optimize timesteps for gdma and tpu operations")
        << "\n";
  });
  status = time_step->assignTimeStep(lg_info, shape_secs, true);
  if (status == false) {
    PROFILE_LOG("is_layer_group_valid", false);
    lg_info.is_valid = NOT_VALID;
    lg_info.group_cost = MAX_COST;
    set_layer_group_cache(lg_info);
    GROUP_DEBUG_WITH_TYPE("lg_result", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "assignTimeStep", "failed",
                          "failed to backward `slice_info` of tensors "
                          "according to `init shape_secs`")
                   << "\n";
    });
    return false;
  }

  auto lmem_allocator = std::make_shared<LmemAllocator>(options_);
  GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "assignLmemAddrWithSecs", "call_function",
                        "try to find a valid `shape_secs` which allows all "
                        "tensors get available "
                        "local memory address and try to make the group reach "
                        "the best performance")
                 << "\n";
  });
  status = lmem_allocator->assignLmemAddrWithSecs(lg_info, time_step,
                                                  shape_secs, false, true);
  // allow bank conflict
  // if (status == false) {
  //   status =
  //     lmem_allocator->assignLmemAddrWithSecs(lg_info, time_step, shape_secs,
  //     true);
  // }
  if (status == false) {
    PROFILE_LOG("is_layer_group_valid", false);
    lg_info.is_valid = NOT_VALID;
    lg_info.group_cost = MAX_COST;
    set_layer_group_cache(lg_info);
    GROUP_DEBUG_WITH_TYPE("lg_result", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "assignLmemAddrWithSecs", "failed",
                          "failed to find a valid `shape_secs` which allows "
                          "all tensors get available "
                          "local memory address")
                   << "\n";
    });
    return false;
  }

  *group_cost = lmem_allocator->get_min_group_cost();
  //   if (calc_cost) {
  // // remove it after pid_node is extractedb
  // #pragma omp critical(get_cycle)
  //     *group_cost =
  //         cycle_calculator_->getGroupCycle(time_step, shape_secs,
  //         lg_info.type);
  //   }
  PROFILE_LOG("is_layer_group_valid", false);
  lg_info.is_valid = VALID;
  lg_info.shape_secs = shape_secs;
  lg_info.group_cost = group_cost ? *group_cost : -1;
  set_layer_group_cache(lg_info);
  GROUP_DEBUG_WITH_TYPE("lg_result", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO("check_is_layer_group_valid",
                                          "success",
                                          "group with multiple layers is valid")
                 << "\n";
  });
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
    const std::vector<Operation *> &base_group, int group_idx,
    int64_t idx_offset) {
  LgInfo sub_group;
  size_t group_layer_num = base_group.size();
  const int64_t max_cluster_size = get_max_cluster_size(group_layer_num);
  // const int64_t max_cluster_size = 1;
  int64_t start_idx = 0, end_idx = 1, cluster_size = 1;

  if (max_cluster_size == 1) {
    for (size_t layer_idx = 0; layer_idx < group_layer_num; ++layer_idx) {
      clusters.push_back(std::make_pair<int64_t, int64_t>(layer_idx, 1));
    }
  } else {
    int64_t pre_cost = 0;
    for (size_t idx = 1; idx < group_layer_num; ++idx) {
      if (start_idx == end_idx - 1) {
        pre_cost = cycle_calculator_->getGlobalLayerCycle(base_group[start_idx])
                       .cycles;
        LG_DEBUG_WITH_TYPE("group_clusters", [&]() {
          llvm::dbgs()
              << DEBUGGER_DEFAULT_INFO(
                     "get_pre_cost", "result",
                     "calculate pre_cost when start_idx == end_idx - 1")
              << LOG_KV("base_group_idx", group_idx)
              << LOG_KV("cost_type", "Global") << LOG_KV("start_idx", start_idx)
              << LOG_KV("end_idx", start_idx) << LOG_KV("cost", pre_cost)
              << "\n";
        });
      }
      int64_t post_cost =
          cycle_calculator_->getGlobalLayerCycle(base_group[end_idx]).cycles;
      LG_DEBUG_WITH_TYPE("group_clusters", [&]() {
        llvm::dbgs() << DEBUGGER_DEFAULT_INFO("get_post_cost", "result",
                                              "calculate post_cost")
                     << LOG_KV("base_group_idx", group_idx)
                     << LOG_KV("cost_type", "Global")
                     << LOG_KV("start_idx", end_idx)
                     << LOG_KV("end_idx", end_idx) << LOG_KV("cost", post_cost)
                     << "\n";
      });

      pre_cost = cost_add(pre_cost, post_cost);

      int64_t temp_cost = 0;
      get_layer_group(sub_group, base_group, start_idx, end_idx, group_idx,
                      idx_offset);
      bool is_valid = is_layer_group_valid(sub_group, true, &temp_cost);

      LG_DEBUG_WITH_TYPE("group_clusters", [&]() {
        llvm::dbgs() << DEBUGGER_DEFAULT_INFO("get_group_cost", "result",
                                              "calculate group cost")
                     << LOG_KV("base_group_idx", group_idx)
                     << LOG_KV("cost_type", "Group")
                     << LOG_KV("start_idx", start_idx)
                     << LOG_KV("end_idx", end_idx) << LOG_KV("cost", temp_cost)
                     << "\n";
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
  LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::outs() << "\n"; });

  LG_DEBUG_WITH_TYPE("cluster_info", [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO("get_group_clusters", "result",
                                          "get clusters info")
                 << "\n";
    for (size_t i = 0; i < clusters.size(); ++i) {
      llvm::dbgs() << LOG_ACTION("cluster_info") << LOG_KV("cluster_idx", i)
                   << LOG_KV("cluster_size", clusters[i].second)
                   << LOG_KV("start_idx", clusters[i].first)
                   << LOG_KV("end_idx",
                             clusters[i].first + clusters[i].second - 1)
                   << "\n";
    }
  });
}

void GroupMethod::get_group_clusters_with_dynamic_programming(
    std::vector<std::pair<int64_t, int64_t>> &clusters,
    const std::vector<Operation *> &base_group, int group_idx,
    int64_t idx_offset, int64_t max_cluster_size) {
  auto &lg_config = LgConfig::getInstance();
  clusters.clear();
  LgInfo lg_info;
  size_t group_layer_num = base_group.size();
  // const int64_t max_cluster_size = get_max_cluster_size(group_layer_num);

  if (max_cluster_size == 1) {
    for (size_t layer_idx = 0; layer_idx < group_layer_num; ++layer_idx) {
      clusters.push_back(std::make_pair<int64_t, int64_t>(layer_idx, 1));
    }
    return;
  }

  auto calc_group_cost = [&](LgInfo &lg_info) {
    int64_t group_cost = MAX_COST;
    is_layer_group_valid(lg_info, true, &group_cost);
    return group_cost;
  };

  auto calc_group_cost_with_cache = [&](LgInfo &lg_info) {
    int64_t group_cost = MAX_COST;
    shape_secs_t shape_secs;
    uint64_t hash_key;
    LgCache::getInstance().get_graph_hash(lg_info, hash_key, false);
    bool cache_hit = false;
#pragma omp critical(layer_group_cost_cache)
    cache_hit = LgCache::getInstance().get_cost_from_cost_cache(
        hash_key, group_cost, lg_info.shape_secs_search_level);
    if (!cache_hit) {
      // if
      // (LgDebugger::getInstance().is_conditional_debug_group(lg_info.func_start_idx,
      //                                           lg_info.func_end_idx)) {
      //   lg_info.dump();
      // }
      group_cost = calc_group_cost(lg_info);
      if (group_cost != MAX_COST) {
        shape_secs = lg_info.shape_secs;
      }
#pragma omp critical(layer_group_cost_cache)
      LgCache::getInstance().add_cost_cache(hash_key, lg_info);
    }
    return group_cost;
  };

  LAYER_GROUP_LOG_DEBUG_BLOCK(
      { llvm::outs() << "Getting clusters using dynamic programming...\n"; });
  LAYER_GROUP_LOG_DEBUG_BLOCK(
      { llvm::outs() << "Getting single group cost...\n"; });
  auto single_group_costs = std::vector<std::vector<int64_t>>(
      group_layer_num, std::vector<int64_t>(group_layer_num, MAX_COST));
  auto shape_secs_search_strategy =
      lg_config.get_config_value<int>("shape_secs_search_strategy", 0);
  progressbar single_group_bar(max_cluster_size - 1);
#pragma omp parallel for private(lg_info) schedule(dynamic, 1)
  for (size_t len = 2; len <= max_cluster_size; ++len) {
    if (options_.debugger < 4) {
#pragma omp critical(flush_progress_bar)
      single_group_bar.update();
    }
    for (int64_t start_idx = 0; start_idx <= group_layer_num - len;
         ++start_idx) {
      int64_t end_idx = start_idx + len - 1;
      get_layer_group(lg_info, base_group, start_idx, end_idx, group_idx,
                      idx_offset);
      if (shape_secs_search_strategy == SHAPE_SECS_ALWAYS_BETTER) {
        lg_info.shape_secs_search_level = 1;
      }
      int64_t cost = LgCache::getInstance().cache_enabled
                         ? calc_group_cost_with_cache(lg_info)
                         : calc_group_cost(lg_info);
      single_group_costs[start_idx][end_idx] = cost;
    }
  }
  llvm::outs() << "\n";

  auto cost_table = std::vector<std::vector<int64_t>>(
      group_layer_num, std::vector<int64_t>(group_layer_num, 0));
  auto cut_points = std::vector<std::vector<int64_t>>(
      group_layer_num, std::vector<int64_t>(group_layer_num, 0));
  for (size_t i = 0; i < group_layer_num; ++i) {
    cost_table[i][i] =
        cycle_calculator_->getGlobalLayerCycle(base_group[i]).cycles;
    cut_points[i][i] = i;
  }
  progressbar cost_table_bar(group_layer_num - 1);
  LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::outs() << "Getting cost_table...\n"; });
  // sweep and update cost_table and cut_points.
  for (size_t len = 2; len <= group_layer_num; ++len) {
    if (options_.debugger < 4) {
      cost_table_bar.update();
    }
    for (int64_t start = 0; start <= group_layer_num - len; ++start) {
      int64_t end = start + len - 1;
      int64_t cost = single_group_costs[start][end];

      int64_t optimal_cut_point = end;
      for (int64_t sweep = start; sweep < end; ++sweep) {
        int64_t temp_cost =
            cost_add(cost_table[start][sweep], cost_table[sweep + 1][end]);
        if (temp_cost < cost) {
          cost = temp_cost;
          optimal_cut_point = sweep;
        }
      }
      cost_table[start][end] = cost;
      cut_points[start][end] = optimal_cut_point;
    }
  }
  llvm::outs() << "\n";
  // extract clusters from cut_points and cost_table.
  std::function<void(int64_t, int64_t)> extract_clusters =
      [&clusters, &cut_points, &extract_clusters](int64_t start, int64_t end) {
        int64_t opt_cut = cut_points[start][end];
        if (opt_cut != end) {
          extract_clusters(start, opt_cut);
          extract_clusters(opt_cut + 1, end);
        } else {
          clusters.push_back(std::make_pair(start, end - start + 1));
        }
      };

  extract_clusters(0, group_layer_num - 1);
  std::sort(
      clusters.begin(), clusters.end(),
      [](const std::pair<int64_t, int64_t> &a,
         const std::pair<int64_t, int64_t> &b) { return a.first < b.first; });
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

bool GroupMethod::dynamic_programming_with_structure_detect(
    const std::vector<Operation *> &base_group, int group_idx,
    int64_t idx_offset) {
  // detect repeated structures from base_group
  // consider base group has repeated structures for several times
  // Remark:
  // 1.only support detecting one structure now
  // 2.two structures should be continuous

  // settings
  int64_t MIN_STRUCTURE_SIZE = 10;

  LgInfo lg_info;
  auto &lg_cache = LgCache::getInstance();
  size_t group_layer_num = base_group.size();
  auto hash_counts = lg_cache.get_hash_counts();
  auto hash_counts_map = lg_cache.get_hash_counts_map();
  auto hash_op_map = lg_cache.get_hash_op_map();
  bool valid = false;
  int64_t structure_size = -1;
  int64_t anchor_op_repeated_times = -1;

  // 1.check whether cnt can be considered as repeated times
  //    a.the structure size should be same
  //    b.the structures should be continuous
  //    c.the structures don't need to cover all layers of base_group
  for (auto iter : hash_counts_map) {
    auto cnt = iter.first;
    auto op_hash_set = iter.second;
    if (cnt <= 2) {
      continue;
    }
    LG_DEBUG_WITH_TYPE("structure_detect", [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "dynamic_programming_with_structure_detect",
                          "iteration", "try to detect repeated anchor op")
                   << LOG_KV("base_group_idx", group_idx)
                   << LOG_KV("group_layer_num", group_layer_num)
                   << LOG_KV("cnt", cnt) << "\n";
    });
    for (auto op_hash : op_hash_set) {
      auto op_locations = hash_op_map[op_hash];
      assert(op_locations.size() == cnt);
      structure_size = op_locations[1] - op_locations[0];
      if (structure_size < MIN_STRUCTURE_SIZE ||
          structure_size * (cnt - 2) > group_layer_num) {
        LG_DEBUG_WITH_TYPE("structure_detect", [&]() {
          llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                              "dynamic_programming_with_structure_detect",
                              "iteration", "structure size judgement")
                       << LOG_KV("result", "invalid")
                       << LOG_KV("op_hash", op_hash)
                       << LOG_KV("structure_size", structure_size)
                       << LOG_KV("cnt", cnt) << "\n";
        });
        continue;
      }
      valid = true;
      anchor_op_repeated_times = cnt;
      LG_DEBUG_WITH_TYPE("structure_detect", [&]() {
        llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                            "dynamic_programming_with_structure_detect",
                            "iteration", "structure size judgement")
                     << LOG_KV("result", "valid") << LOG_KV("op_hash", op_hash)
                     << LOG_KV("structure_size", structure_size)
                     << LOG_KV("anchor_op_repeated_times",
                               anchor_op_repeated_times)
                     << "\n";
      });
      break;
    }
    if (valid) {
      break;
    }
  }

  if (valid) {
    auto structure_start_idx = -1;
    // 2.find the start location of first structure using sliding window
    //   try to find continuous structures repeated at least 3 times
    for (int64_t start_idx = 0; start_idx <= group_layer_num; start_idx += 1) {
      bool match = true;
      uint64_t stucture_hash;
      int64_t MIN_REPEATED_TIMES = 3;
      for (int64_t t = 0; t < MIN_REPEATED_TIMES; ++t) {
        auto st = start_idx + t * structure_size;
        auto ed = st + structure_size - 1;
        if (!get_layer_group(lg_info, base_group, st, ed, group_idx,
                             idx_offset)) {
          match = false;
          break;
        }
        uint64_t temp_hash;
        lg_cache.get_graph_hash(lg_info, temp_hash, true);
        if (t == 0) {
          stucture_hash = temp_hash;
        } else {
          if (stucture_hash != temp_hash) {
            match = false;
            break;
          }
        }
      }
      if (match) {
        structure_start_idx = start_idx;
        break;
      }
    }
    if (structure_start_idx < 0) {
      LG_DEBUG_WITH_TYPE("structure_detect", [&]() {
        llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                            "dynamic_programming_with_structure_detect",
                            "structure_start_idx",
                            "failed to find the start idx of first structure")
                     << LOG_KV("base_group_idx", group_idx)
                     << LOG_KV("group_layer_num", group_layer_num)
                     << LOG_KV("idx_offset", idx_offset)
                     << LOG_KV("structure_size", structure_size) << "\n";
      });
      return false;
    }
    LG_DEBUG_WITH_TYPE("structure_detect", [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "dynamic_programming_with_structure_detect",
                          "structure_start_idx",
                          "find the start idx of first structure")
                   << LOG_KV("base_group_idx", group_idx)
                   << LOG_KV("group_layer_num", group_layer_num)
                   << LOG_KV("idx_offset", idx_offset)
                   << LOG_KV("structure_size", structure_size)
                   << LOG_KV("structure_start_idx", structure_start_idx)
                   << "\n";

      if (get_layer_group(lg_info, base_group, structure_start_idx,
                          structure_start_idx + structure_size - 1, group_idx,
                          idx_offset)) {
        lg_info.dump();
      }
    });
    // 3.put two same structures together as a base group
    //   use dynamic programming with cluster to get patterns from this base
    //   group sort patterns according to their lengths from long to short use
    //   these patterns to match the whole base_group
    auto double_structure_group_size = 2 * structure_size;
    std::vector<std::pair<int64_t, int64_t>> double_structure_clusters;
    std::vector<Operation *> double_structure_base_group(
        base_group.begin() + structure_start_idx,
        base_group.begin() + structure_start_idx + double_structure_group_size);
    std::vector<int64_t> structure_cut_result;
    dynamic_programming_with_cluster(
        double_structure_base_group, double_structure_clusters,
        structure_cut_result, group_idx, idx_offset + structure_start_idx,
        get_max_cluster_size(double_structure_group_size));
    LG_DEBUG_WITH_TYPE("structure_detect", [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "dynamic_programming_with_structure_detect",
                          "dynamic_programming_with_cluster",
                          "get cut_result of double_structure_base_group")
                   << "\n";
      int start = 0;
      for (auto end : structure_cut_result) {
        get_layer_group(lg_info, double_structure_base_group, start, end, -1,
                        idx_offset + structure_start_idx);
        int64_t group_cost = MAX_COST;
        auto temp_status = is_layer_group_valid(lg_info, true, &group_cost);
        llvm::dbgs() << temp_status << " ;start[" << start << "] -> end[" << end
                     << "]; group_cost = " << group_cost << "\n";
        lg_info.dump();
        start = end + 1;
      }
      llvm::dbgs() << "double_structure_base_group cut results: ";
      for (size_t j = 0; j < structure_cut_result.size(); ++j) {
        llvm::dbgs() << structure_cut_result[j] << ", ";
      }
      llvm::dbgs() << "\n";
    });

    if (structure_cut_result.size() > 2) {
      // 4.a.double_structure_base_group is cut into N patterns (N > 2)
      //     drop the first and the last patterns which may be incomplete
      //     use the middle patterns to match the whole base_group
      std::set<uint64_t> pattern_set;
      std::set<uint64_t, std::greater<uint64_t>> pattern_len_set;
      for (size_t i = 1; i + 1 < structure_cut_result.size(); ++i) {
        int64_t pattern_start = structure_cut_result[i - 1] + 1;
        int64_t pattern_end = structure_cut_result[i];
        int64_t pattern_size = pattern_end - pattern_start + 1;
        uint64_t pattern_hash;
        get_layer_group(lg_info, double_structure_base_group, pattern_start,
                        pattern_end, -1, idx_offset + structure_start_idx);
        lg_cache.get_graph_hash(lg_info, pattern_hash, true);
        pattern_set.insert(pattern_hash);
        pattern_len_set.insert(pattern_size);
        LG_DEBUG_WITH_TYPE("structure_detect", [&]() {
          llvm::dbgs()
              << DEBUGGER_DEFAULT_INFO(
                     "dynamic_programming_with_structure_detect",
                     "pattern_info",
                     "extract pattern from double_structure_base_group")
              << LOG_KV("base_group_idx", group_idx) << LOG_KV("pattern_idx", i)
              << LOG_KV("pattern_start_idx",
                        pattern_start + structure_start_idx)
              << LOG_KV("pattern_end_idx", pattern_end + structure_start_idx)
              << LOG_KV("pattern_size", pattern_size)
              << LOG_KV("pattern_hash", pattern_hash) << "\n";
          lg_info.dump();
        });
      }
      // 5.match the whole base_group using the patterns
      std::vector<std::pair<int64_t, int64_t>> pattern_clusters;
      std::vector<bool> is_covered(base_group.size(), false);
      for (auto pattern_len : pattern_len_set) {
        for (int i = 0; i + pattern_len - 1 < group_layer_num;) {
          get_layer_group(lg_info, base_group, i, i + pattern_len - 1,
                          group_idx, idx_offset);
          uint64_t temp_hash;
          lg_cache.get_graph_hash(lg_info, temp_hash, true);
          bool match = pattern_set.find(temp_hash) != pattern_set.end();
          if (match && !is_covered[i] && !is_covered[i + pattern_len - 1]) {
            pattern_clusters.push_back(
                std::make_pair<int64_t, int64_t>(i, pattern_len));
            for (int64_t j = 0; j < pattern_len; ++j) {
              is_covered[i + j] = true;
            }
            i += pattern_len;
          } else {
            i += 1;
          }
        }
      }
      std::sort(pattern_clusters.begin(), pattern_clusters.end(),
                [](const std::pair<int64_t, int64_t> &a,
                   const std::pair<int64_t, int64_t> &b) {
                  return a.first < b.first;
                });
      LG_DEBUG_WITH_TYPE("structure_detect", [&]() {
        llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                            "dynamic_programming_with_structure_detect",
                            "pattern_clusters",
                            "show pattern_clusters after structure detection")
                     << LOG_KV("base_group_idx", group_idx)
                     << LOG_KV("num_clusters", pattern_clusters.size()) << "\n";
        for (size_t i = 0; i < pattern_clusters.size(); ++i) {
          llvm::dbgs() << LOG_KV("pattern_cluster_idx", i)
                       << LOG_KV("start_idx", pattern_clusters[i].first)
                       << LOG_KV("end_idx", pattern_clusters[i].first +
                                                pattern_clusters[i].second - 1)
                       << LOG_KV("cluster_size", pattern_clusters[i].second)
                       << "\n";
        }
      });
      int64_t start_idx = 0;
      std::vector<int64_t> cut_result;
      auto pattern_clusters_size = pattern_clusters.size();
      for (int64_t i = 0; i < pattern_clusters_size; ++i) {
        int64_t cluster_start_idx = pattern_clusters[i].first;
        int64_t cluster_end_idx =
            pattern_clusters[i].first + pattern_clusters[i].second - 1;
        cut_result.push_back(cluster_end_idx); // add current pattern cluster
        LG_DEBUG_WITH_TYPE("structure_detect", [&]() {
          llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                              "dynamic_programming_with_structure_detect",
                              "tmp_cut_result",
                              "cut results from pattern_clusters")
                       << LOG_KV("base_group_idx", group_idx)
                       << LOG_KV("cluster_start_idx", cluster_start_idx)
                       << LOG_KV("cluster_end_idx", cluster_end_idx) << "\n";
        });
        bool do_dp = false;
        int64_t dp_start_idx = -1;
        int64_t dp_end_idx = -1;
        if (cluster_start_idx > start_idx) {
          do_dp = true;
          dp_start_idx = start_idx;
          dp_end_idx = cluster_start_idx - 1;
        } else if (i == pattern_clusters_size - 1 &&
                   cluster_end_idx < group_layer_num - 1) {
          do_dp = true;
          dp_start_idx = cluster_end_idx + 1;
          dp_end_idx = group_layer_num - 1;
        }
        start_idx = cluster_end_idx + 1;
        if (do_dp) {
          std::vector<Operation *> non_structure_base_group(
              base_group.begin() + dp_start_idx,
              base_group.begin() + dp_end_idx + 1);
          std::vector<std::pair<int64_t, int64_t>> tmp_clusters;
          std::vector<int64_t> tmp_cut_result;
          dynamic_programming_with_cluster(
              non_structure_base_group, tmp_clusters, tmp_cut_result, group_idx,
              idx_offset + dp_start_idx, 1);
          for (auto &idx : tmp_cut_result) {
            idx += dp_start_idx;
          }
          LG_DEBUG_WITH_TYPE("structure_detect", [&]() {
            llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                                "dynamic_programming_with_structure_detect",
                                "non_structure_base_group",
                                "get cut_result of non_structure_base_group")
                         << "\n";
            int start = dp_start_idx;
            for (auto end : tmp_cut_result) {
              auto tmp_start = start - dp_start_idx;
              auto tmp_end = end - dp_start_idx;
              get_layer_group(lg_info, non_structure_base_group, tmp_start,
                              tmp_end, -1, idx_offset + dp_start_idx);
              int64_t group_cost = MAX_COST;
              auto temp_status =
                  is_layer_group_valid(lg_info, true, &group_cost);
              llvm::dbgs() << temp_status << " ;start[" << start << "] -> end["
                           << end << "]; group_cost = " << group_cost << "\n";
              lg_info.dump();
              start = end + 1;
            }
            llvm::dbgs() << "non-structure regions cut results: ";
            for (size_t j = 0; j < tmp_cut_result.size(); ++j) {
              llvm::dbgs() << tmp_cut_result[j] << ", ";
            }
            llvm::dbgs() << "\n";
          });
          cut_result.insert(cut_result.end(), tmp_cut_result.begin(),
                            tmp_cut_result.end());
        }
      }
      std::sort(cut_result.begin(), cut_result.end());
      LG_DEBUG_WITH_TYPE("structure_detect", [&]() {
        llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                            "dynamic_programming_with_structure_detect",
                            "final_cut_result",
                            "final cut results after structure detection")
                     << LOG_KV("base_group_idx", group_idx)
                     << LOG_KV("num_cut_points", cut_result.size()) << "\n";
        llvm::dbgs() << "base_gorup cut results: ";
        for (size_t j = 0; j < cut_result.size(); ++j) {
          llvm::dbgs() << cut_result[j] << ", ";
        }
        llvm::dbgs() << "\n";
      });
      cut_results_.push_back(std::move(cut_result));
    } else {
      std::vector<std::pair<int64_t, int64_t>> base_group_clusters;
      std::vector<int64_t> cut_result;
      dynamic_programming_with_cluster(base_group, base_group_clusters,
                                       cut_result, group_idx, idx_offset,
                                       structure_size);
      LG_DEBUG_WITH_TYPE("structure_detect", [&]() {
        llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                            "dynamic_programming_with_structure_detect",
                            "final_cutresult",
                            "final cut results after structure detection")
                     << LOG_KV("base_group_idx", group_idx)
                     << LOG_KV("num_cut_points", cut_result.size()) << "\n";
        llvm::dbgs() << "base_gorup cut results: ";
        for (size_t j = 0; j < cut_result.size(); ++j) {
          llvm::dbgs() << cut_result[j] << ", ";
        }
        llvm::dbgs() << "\n";
      });
      cut_results_.push_back(std::move(cut_result));
    }
    return true;
  }
  return false;
}

void GroupMethod::dynamic_programming_kernel(
    LgInfo &lg_info, const std::vector<Operation *> &base_group,
    const std::vector<std::pair<int64_t, int64_t>> &clusters,
    std::vector<std::vector<int64_t>> &cost_table,
    std::vector<std::vector<int64_t>> &cut_points, int64_t base_group_idx,
    int64_t idx_offset) {
  auto &lg_debugger = LgDebugger::getInstance();
  auto &lg_config = LgConfig::getInstance();
  auto cluster_num = clusters.size();
  // auto cost_table = std::vector<std::vector<int64_t>>(
  //     cluster_num, std::vector<int64_t>(cluster_num, 0));
  // auto cut_points = std::vector<std::vector<int64_t>>(
  //     cluster_num, std::vector<int64_t>(cluster_num, 0));
  auto shape_secs_search_strategy =
      lg_config.get_config_value<int>("shape_secs_search_strategy", 0);
  for (size_t j = 0; j < cluster_num; ++j) {
    int64_t start_idx = clusters[j].first;
    int64_t end_idx = start_idx + clusters[j].second - 1;
    get_layer_group(lg_info, base_group, start_idx, end_idx, base_group_idx,
                    idx_offset);

    if (shape_secs_search_strategy == SHAPE_SECS_ALWAYS_BETTER) {
      lg_info.shape_secs_search_level = 1;
    }
    assert(is_layer_group_valid(lg_info, true, &cost_table[j][j]));

    GROUP_DEBUG_WITH_TYPE("lg_cost", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO("cluster_cost", "record",
                                            "calculate cost_table[%d][%d]", j,
                                            j)
                   << LOG_KV("base_group_idx", base_group_idx)
                   << LOG_KV("start_idx", lg_info.start_idx)
                   << LOG_KV("end_idx", lg_info.end_idx)
                   << LOG_KV("func_start_idx", lg_info.func_start_idx)
                   << LOG_KV("func_end_idx", lg_info.func_end_idx)
                   << LOG_KV("cost", cost_table[j][j]) << "\n";
    });

    cut_points[j][j] = j;
  }

  /**
   * you can debug any cluster like calc_cost(16, 17);
   */
  auto calc_group_cost = [&](LgInfo &lg_info) {
    GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "is_layer_group_valid", "call_function",
                          "check if the group is valid and calculate the cost")
                   << "\n";
    });
    // if (lg_debugger.is_conditional_debug_group(sub_group.func_start_idx,
    // sub_group.func_end_idx)) {
    //   return MAX_COST;
    // }
    int64_t group_cost = MAX_COST;
    auto is_valid = is_layer_group_valid(lg_info, true, &group_cost);
    if (options_.debugger > 2 && is_valid) {
      int64_t manual_group_cost = lg_debugger.get_manual_group_cost(lg_info);
      if (manual_group_cost != -1) { // -1 means no manual cost
        return manual_group_cost;
      }
    }
    return group_cost;
  };

  auto calc_group_cost_with_cache = [&](LgInfo &lg_info) {
    int64_t group_cost = MAX_COST;
    shape_secs_t shape_secs;
    uint64_t hash_key;
    LgCache::getInstance().get_graph_hash(lg_info, hash_key, false);
    bool cache_hit = false;
#pragma omp critical(layer_group_cost_cache)
    cache_hit = LgCache::getInstance().get_cost_from_cost_cache(
        hash_key, group_cost, lg_info.shape_secs_search_level);
    if (options_.debugger > 2 && cache_hit) {
      int64_t manual_group_cost = lg_debugger.get_manual_group_cost(lg_info);
      if (manual_group_cost != -1) { // -1 means no manual cost
        lg_info.shape_secs = {};
        lg_info.shape_secs_search_level = 0;
        LgCache::getInstance().add_cost_cache(hash_key, lg_info);
        return manual_group_cost;
      }
    }
    if (!cache_hit) {
      group_cost = calc_group_cost(lg_info);
      if (group_cost != MAX_COST) {
        shape_secs = lg_info.shape_secs;
      }
#pragma omp critical(layer_group_cost_cache)
      LgCache::getInstance().add_cost_cache(hash_key, lg_info);
    }
    return group_cost;
  };

  LAYER_GROUP_LOG_DEBUG_BLOCK(
      { llvm::outs() << "Searching best group slices...\n"; });
  LAYER_GROUP_LOG_DEBUG_BLOCK(
      { llvm::outs() << "Getting single group cost...\n"; });
  progressbar single_group_bar(cluster_num - 1);
  auto single_group_costs = std::vector<std::vector<int64_t>>(
      cluster_num, std::vector<int64_t>(cluster_num, MAX_COST));
#pragma omp parallel for private(lg_info) schedule(dynamic, 1)
  for (size_t len = 2; len <= cluster_num; ++len) {
    if (options_.debugger < 4) {
#pragma omp critical(flush_progress_bar)
      single_group_bar.update();
    }
    for (int64_t start = 0; start <= cluster_num - len; ++start) {
      int64_t end = start + len - 1;
      int64_t start_idx = clusters[start].first;
      int64_t end_idx = clusters[end].first + clusters[end].second - 1;
      get_layer_group(lg_info, base_group, start_idx, end_idx, base_group_idx,
                      idx_offset);
      if (shape_secs_search_strategy == SHAPE_SECS_ALWAYS_BETTER) {
        lg_info.shape_secs_search_level = 1;
      }
      int64_t cost = LgCache::getInstance().cache_enabled
                         ? calc_group_cost_with_cache(lg_info)
                         : calc_group_cost(lg_info);
      single_group_costs[start][end] = cost;
      GROUP_DEBUG_WITH_TYPE("group_cost", lg_info, [&]() {
        llvm::dbgs()
            << DEBUGGER_DEFAULT_INFO(
                   "single_group_costs", "record",
                   "calculate single_group_costs(start_idx=%d, end_idx=%d)",
                   start_idx, end_idx)
            << LOG_KV("func_start_idx", lg_info.func_start_idx)
            << LOG_KV("func_end_idx", lg_info.func_end_idx)
            << LOG_KV("cost", cost)
            << LOG_KV_FORMAT("shape_secs", "%d,%d,%d,%d,%d",
                             lg_info.shape_secs.nsecs, lg_info.shape_secs.csecs,
                             lg_info.shape_secs.dsecs, lg_info.shape_secs.hsecs,
                             lg_info.shape_secs.wsecs)
            << "\n";
      });
    }
  }
  llvm::outs() << "\n";

  progressbar cost_table_bar(cluster_num - 1);
  LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::outs() << "Getting cost table...\n"; });
  // sweep and update cost_table and cut_points.
  for (size_t len = 2; len <= cluster_num; ++len) {
    if (options_.debugger < 4) {
      cost_table_bar.update();
    }
    for (int64_t start = 0; start <= cluster_num - len; ++start) {
      int64_t end = start + len - 1;
      int64_t cost = single_group_costs[start][end];
      int64_t group_cost = cost;

      int64_t optimal_cut_point = end;
      for (int64_t sweep = start; sweep < end; ++sweep) {
        int64_t temp_cost =
            cost_add(cost_table[start][sweep], cost_table[sweep + 1][end]);
        GROUP_DEBUG_WITH_TYPE("cost_sweep", lg_info, [&]() {
          llvm::dbgs()
              << DEBUGGER_DEFAULT_INFO(
                     "interval_cost", "record",
                     "calculate (cost_table[%d][%d] + cost_table[%d][%d])",
                     start, sweep, sweep + 1, end)
              << LOG_KV("idx_offset", idx_offset) << LOG_KV("cost", temp_cost)
              << "\n";
        });
        if (temp_cost < cost) {
          cost = temp_cost;
          optimal_cut_point = sweep;
        }
      }

      int64_t start_idx = clusters[start].first;
      int64_t end_idx = clusters[end].first + clusters[end].second - 1;
      get_layer_group(lg_info, base_group, start_idx, end_idx, base_group_idx,
                      idx_offset);
      if (shape_secs_search_strategy == SHAPE_SECS_BETTER_IF_MIGHT_SPLIT) {
        // try to find whether group cost can be reduced
        // TODO: judgment group_cost != MAX_COST have to be removed after
        if (optimal_cut_point != end && group_cost != MAX_COST) {
          lg_info.shape_secs_search_level = 1;
          int64_t group_cost_opt = LgCache::getInstance().cache_enabled
                                       ? calc_group_cost_with_cache(lg_info)
                                       : calc_group_cost(lg_info);
          if (group_cost_opt < cost) {
            single_group_costs[start][end] = group_cost_opt;
            cost = group_cost_opt;
            optimal_cut_point = end;
          }
        }
      }

      cost_table[start][end] = cost;
      cut_points[start][end] = optimal_cut_point;
      GROUP_DEBUG_WITH_TYPE("cost_table", lg_info, [&]() {
        llvm::dbgs() << DEBUGGER_DEFAULT_INFO("cost_table", "record",
                                              "calculate cost_table[%d][%d]",
                                              start, end)
                     << LOG_KV("optimal_cut_point", optimal_cut_point)
                     << LOG_KV("func_start_idx", lg_info.func_start_idx)
                     << LOG_KV("func_end_idx", lg_info.func_end_idx)
                     << LOG_KV("cost", cost) << "\n";
      });
    }
  }
  // GROUP_DEBUG_WITH_TYPE("lg_cost", lg_info, [&]() {
  //   for (size_t len = 2; len <= cluster_num; ++len) {
  //     for (int64_t start = 0; start <= cluster_num - len; ++start) {
  //       int64_t end = start + len - 1;
  //       int64_t optimal_cut_point = cut_points[start][end];
  //       int64_t start_idx = clusters[start].first;
  //       int64_t end_idx = clusters[end].first + clusters[end].second - 1;
  //       get_layer_group(lg_info, base_group, start_idx, end_idx,
  //       base_group_idx,
  //                       idx_offset);
  //       llvm::dbgs() << DEBUGGER_DEFAULT_INFO("cost_table", "record",
  //                                             "calculate cost_table[%d][%d]",
  //                                             start, end)
  //                   << LOG_KV("optimal_cut_point", optimal_cut_point)
  //                   << LOG_KV("func_start_idx", lg_info.func_start_idx)
  //                   << LOG_KV("func_end_idx", lg_info.func_end_idx)
  //                   << LOG_KV("cost", cost_table[start][end]) << "\n";
  //     }
  //   }
  // });

  llvm::outs() << "\n";
  // std::vector<int64_t> cut_result;
  // get_layer_cut_result(cut_result, clusters, cut_points, 0, cluster_num - 1);
  // cut_results_.push_back(std::move(cut_result));
  // LLVM_DEBUG({
  //   LgInfo lg_info;
  //   int start = 0;
  //   for (auto end : cut_result) {
  //     get_layer_group(lg_info, base_group, start, end, base_group_idx,
  //                     idx_offset);
  //     int64_t group_cost = MAX_COST;
  //     auto temp_status = is_layer_group_valid(lg_info, true, &group_cost);
  //     llvm::dbgs() << temp_status << " ;start" << start << " - " << " end "
  //                  << end << " = " << group_cost << "\n";
  //     start = end + 1;
  //   }

  //   llvm::dbgs() << "\n";
  //   llvm::dbgs() << "================FINAL GROUP================\n";
  //   for (size_t cost_i = 0; cost_i < cluster_num; ++cost_i) {
  //     for (int64_t cost_j = 0; cost_j < cluster_num; ++cost_j) {
  //       llvm::dbgs() << cut_points[cost_i][cost_j] << ", " << "";
  //     }
  //     llvm::dbgs() << "\n";
  //   }
  //   llvm::dbgs() << "================COST TABLE================\n";
  //   for (size_t cost_i = 0; cost_i < cluster_num; ++cost_i) {
  //     for (int64_t cost_j = 0; cost_j < cluster_num; ++cost_j) {
  //       llvm::dbgs() << cost_table[cost_i][cost_j] << ", " << "";
  //     }
  //     llvm::dbgs() << "\n";
  //   }
  //   llvm::dbgs() << "=============================================\n";
  //   llvm::dbgs() << "\n";
  // });
}

void GroupMethod::single_group_debug(
    std::vector<LgInfo> &lg_infos,
    const llvm::SetVector<Operation *> &subnet_ops) {
  auto &lg_debugger = LgDebugger::getInstance();
  auto &lg_config = LgConfig::getInstance();
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::dbgs()
        << "\n"
        << "========================================================\n"
        << "**                 Layer group debug                  **\n"
        << "========================================================\n";
  });
  LgInfo lg_info;
  int debug_group_idx = 0;
  for (auto iter : lg_debugger.get_conditional_debug_groups()) {
    LG_DEBUG_WITH_TYPE("lg_step", [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "lg_debugger_iteration_start", "stamp",
                          "process debug_groups[%d], start_idx=%d, end_idx=%d",
                          debug_group_idx, iter.first, iter.second)
                   << "\n";
    });
    auto start_idx = iter.first;
    auto end_idx = iter.second;
    std::vector<Operation *> debug_group;
    std::vector<std::pair<int64_t, int64_t>> clusters;
    auto cluster_num = end_idx - start_idx + 1;
    get_debug_group(debug_group, subnet_ops, start_idx, end_idx);
    get_debug_cluster(clusters, cluster_num);
    LG_DEBUG_WITH_TYPE("lg_step", [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "dynamic_programming_kernel", "call_function",
                          "process clusters using dynamic programming "
                          "algorithm, cluster_num=%d",
                          cluster_num)
                   << "\n";
    });
    /**
     * you can debug any cluster like calc_cost(16, 17);
     */
    auto calc_group_cost = [&](LgInfo &lg_info) {
      GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
        llvm::dbgs()
            << DEBUGGER_DEFAULT_INFO(
                   "is_layer_group_valid", "call_function",
                   "check if the group is valid and calculate the cost")
            << "\n";
      });
      // if (lg_debugger.is_conditional_debug_group(sub_group.func_start_idx,
      // sub_group.func_end_idx)) {
      //   return MAX_COST;
      // }
      int64_t group_cost = MAX_COST;
      auto is_valid = is_layer_group_valid(lg_info, true, &group_cost);
      if (options_.debugger > 2 && is_valid) {
        int64_t manual_group_cost = lg_debugger.get_manual_group_cost(lg_info);
        if (manual_group_cost != -1) { // -1 means no manual cost
          return manual_group_cost;
        }
      }
      return group_cost;
    };
    int64_t start = 0;
    int64_t end = end_idx - start_idx;
    get_layer_group(lg_info, debug_group, start, end, debug_group_idx,
                    start_idx);
    if (lg_config.get_shape_secs_search_strategy() ==
        SHAPE_SECS_ALWAYS_BETTER) {
      lg_info.shape_secs_search_level = 1;
    }
    int64_t cost = calc_group_cost(lg_info);
    GROUP_DEBUG_WITH_TYPE("group_cost", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "single_group_costs", "record",
                          "calculate single_group_costs(func_start_idx=%d, "
                          "func_end_idx=%d)",
                          lg_info.func_start_idx, lg_info.func_end_idx)
                   << LOG_KV("cost", cost)
                   << LOG_KV_FORMAT(
                          "shape_secs", "%d,%d,%d,%d,%d",
                          lg_info.shape_secs.nsecs, lg_info.shape_secs.csecs,
                          lg_info.shape_secs.dsecs, lg_info.shape_secs.hsecs,
                          lg_info.shape_secs.wsecs)
                   << "\n";
    });
  }
}

void GroupMethod::dynamic_programming_layer_group_with_cluster_debug(
    std::vector<LgInfo> &lg_infos,
    const llvm::SetVector<Operation *> &subnet_ops) {
  auto &lg_debugger = LgDebugger::getInstance();
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::dbgs()
        << "\n"
        << "========================================================\n"
        << "** Dynamic Programming layer group with cluster debug **\n"
        << "========================================================\n";
  });
  LgInfo lg_info;
  int debug_group_idx = 0;
  for (auto iter : lg_debugger.get_conditional_debug_groups()) {
    LG_DEBUG_WITH_TYPE("lg_step", [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "lg_debugger_iteration_start", "stamp",
                          "process debug_groups[%d], start_idx=%d, end_idx=%d",
                          debug_group_idx, iter.first, iter.second)
                   << "\n";
    });
    auto start_idx = iter.first;
    auto end_idx = iter.second;
    std::vector<Operation *> debug_group;
    std::vector<std::pair<int64_t, int64_t>> clusters;
    auto cluster_num = end_idx - start_idx + 1;
    get_debug_group(debug_group, subnet_ops, start_idx, end_idx);
    get_debug_cluster(clusters, cluster_num);
    LG_DEBUG_WITH_TYPE("lg_step", [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "dynamic_programming_kernel", "call_function",
                          "process clusters using dynamic programming "
                          "algorithm, cluster_num=%d",
                          cluster_num)
                   << "\n";
    });
    auto cost_table = std::vector<std::vector<int64_t>>(
        cluster_num, std::vector<int64_t>(cluster_num, 0));
    auto cut_points = std::vector<std::vector<int64_t>>(
        cluster_num, std::vector<int64_t>(cluster_num, 0));
    dynamic_programming_kernel(lg_info, debug_group, clusters, cost_table,
                               cut_points, debug_group_idx++, start_idx);
    std::vector<int64_t> cut_result;
    get_layer_cut_result(cut_result, clusters, cut_points, 0, cluster_num - 1);
    cut_results_.push_back(std::move(cut_result));
    LLVM_DEBUG({
      LgInfo lg_info;
      int start = 0;
      for (auto end : cut_result) {
        get_layer_group(lg_info, debug_group, start, end, -1, start_idx);
        int64_t group_cost = MAX_COST;
        auto temp_status = is_layer_group_valid(lg_info, true, &group_cost);
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
  }
}

void GroupMethod::dynamic_programming_with_cluster(
    const std::vector<Operation *> &base_group,
    std::vector<std::pair<int64_t, int64_t>> &clusters,
    std::vector<int64_t> &cut_result, int group_idx, int64_t idx_offset,
    int64_t max_cluster_size) {
  clusters.clear();
  get_group_clusters_with_dynamic_programming(clusters, base_group, group_idx,
                                              idx_offset, max_cluster_size);
  auto cluster_num = clusters.size();
  auto cost_table = std::vector<std::vector<int64_t>>(
      cluster_num, std::vector<int64_t>(cluster_num, 0));
  auto cut_points = std::vector<std::vector<int64_t>>(
      cluster_num, std::vector<int64_t>(cluster_num, 0));
  LgInfo lg_info;
  dynamic_programming_kernel(lg_info, base_group, clusters, cost_table,
                             cut_points, group_idx, idx_offset);
  get_layer_cut_result(cut_result, clusters, cut_points, 0, cluster_num - 1);
  return;
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
  auto &lg_config = LgConfig::getInstance();
  LgCache::getInstance().init(
      subnet_ops,
      runmode_ == RunMode::TPU_DYNAMIC); // To disable cache, comment this line.
  // exit(0);
  LgInfo sub_group;
  std::vector<std::vector<Operation *>> base_groups;
  get_base_groups(base_groups, subnet_ops);
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::outs() << llvm::format("total num of base_group is %d\n",
                                 base_groups.size());
  });
  int64_t idx_offset = 0;
  for (size_t i = 0; i < base_groups.size();
       idx_offset += base_groups[i].size(), ++i) {
    LG_DEBUG_WITH_TYPE("lg_step", [&]() {
      llvm::dbgs()
          << DEBUGGER_DEFAULT_INFO(
                 "base_groups_iteration_start", "stamp",
                 "process base_groups[%d], layer_num=%d, idx_offset=%d", i,
                 base_groups[i].size(), idx_offset)
          << "\n";
    });
    std::vector<std::pair<int64_t, int64_t>> clusters;
    LG_DEBUG_WITH_TYPE("lg_step", [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "get_group_clusters", "call_function",
                          "get group clusters of base_groups[%d]", i)
                   << "\n";
    });
    if (lg_config.get_config_value("structure_detect_opt", true) &&
        !getenv("DISABLE_STRUCTURE_DETECT_OPT")) {
      bool structure_detected = dynamic_programming_with_structure_detect(
          base_groups[i], i, idx_offset);
      if (structure_detected) {
        continue;
      }
    }
    // get_group_clusters(clusters, base_groups[i], i, idx_offset);
    get_group_clusters_with_dynamic_programming(
        clusters, base_groups[i], i, idx_offset,
        get_max_cluster_size(base_groups[i].size()));
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
      LG_DEBUG_WITH_TYPE("lg_step", [&]() {
        llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                            "dynamic_programming_kernel", "call_function",
                            "process clusters using dynamic programming "
                            "algorithm, cluster_num=%d",
                            cluster_num)
                     << "\n";
      });
      dynamic_programming_kernel(sub_group, base_groups[i], clusters,
                                 cost_table, cut_points, i, idx_offset);
      std::vector<int64_t> cut_result;
      get_layer_cut_result(cut_result, clusters, cut_points, 0,
                           cluster_num - 1);
      cut_results_.push_back(std::move(cut_result));
      LLVM_DEBUG({
        LgInfo lg_info;
        int start = 0;
        for (auto end : cut_result) {
          get_layer_group(lg_info, base_groups[i], start, end, i, idx_offset);
          int64_t group_cost = MAX_COST;
          auto temp_status = is_layer_group_valid(lg_info, true, &group_cost);
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
      LG_DEBUG_WITH_TYPE("lg_step", [&]() {
        llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                            "single_cluster", "stamp",
                            "process clusters whose size is 1")
                     << "\n";
      });
      cut_results_.push_back(std::vector<int64_t>(1, 0));
      int64_t start_idx = clusters[0].first;
      get_layer_group(sub_group, base_groups[i], start_idx, start_idx, i,
                      idx_offset);
      GROUP_DEBUG_WITH_TYPE("lg_cost", sub_group, [&]() {
        if (!isa<ReturnOp>(base_groups[i][0]) &&
            runmode_ == RunMode::TPU_STATIC) {
          int64_t cost;
          assert(is_layer_group_valid(sub_group, true, &cost));
          llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                              "single_cluster", "record",
                              "calculate cost of single cluster")
                       << LOG_KV("base_group_idx", i)
                       << LOG_KV("func_start_idx", sub_group.func_start_idx)
                       << LOG_KV("func_end_idx", sub_group.func_end_idx)
                       << LOG_KV("cost", cost) << "\n";
        } else {
          llvm::dbgs()
              << LOG_STEP("GroupMethod::dynamic_programming_layer_group_with_"
                          "cluster@[cost of specific case is set to 0]")
              << DEBUGGER_DEFAULT_INFO("single_cluster", "record",
                                       "cost of specific case is set to 0")
              << LOG_KV("group_idx", i)
              << LOG_KV("func_start_idx", sub_group.func_start_idx)
              << LOG_KV("func_end_idx", sub_group.func_end_idx)
              << LOG_KV("cost", 0) << "\n";
        }
      });
    }
  }

  show_cut_results();
  llvm::dbgs() << "================FINAL GROUP================\n";
  // some post process for cluster
  // LAYER_GROUP_LOG_DEBUG_BLOCK({
  //   llvm::outs() <<
  //   "-------------------------------------------------------\n"; llvm::outs()
  //   << "Consider redundant computation and gdma cost\n"; llvm::outs() <<
  //   "-------------------------------------------------------\n";
  // });
  // consider_redundant_computation_and_gdma_cost(base_groups);
  // show_cut_results();

  // LAYER_GROUP_LOG_DEBUG_BLOCK({
  //   llvm::outs() <<
  //   "-------------------------------------------------------\n"; llvm::outs()
  //   << "Merge cut idx to reduce gdma cost\n"; llvm::outs() <<
  //   "-------------------------------------------------------\n";
  // });
  // bool take_effective = merge_cut_idx_to_reduce_gdma_cost(base_groups);
  // show_cut_results();

  // if (take_effective) {
  //   LAYER_GROUP_LOG_DEBUG_BLOCK({
  //     llvm::outs()
  //         << "-------------------------------------------------------\n";
  //     llvm::outs() << "Consider redundant computation and gdma cost again\n"
  //                  << "due to cut idx merged in the previous step\n";
  //     llvm::outs()
  //         << "-------------------------------------------------------\n";
  //   });
  //   consider_redundant_computation_and_gdma_cost(base_groups);
  //   show_cut_results();
  // }

  // for debug, fix cut results
  // std::vector<int64_t> override_is = {8, 10, 12, 16, 24, 26, 36, 42, 77, 91,
  // 126, 133}; std::vector<int64_t> override_is = {8, 10, 12, 20, 22, 24, 26,
  // 32, 34, 36, 44, 45, 46, 47, 54, 58, 74, 76, 78, 90, 98, 105, 107, 109, 112,
  // 119, 120, 121, 122, 126, 133}; cut_results_[0] = override_is;
  // show_cut_results();

  // update lg_infos
  if (LgCache::getInstance().cache_enabled) {
    get_final_groups_by_cache(lg_infos, base_groups);
  } else {
    get_final_groups(lg_infos, base_groups);
  }

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

#define CALC_FULL
#ifdef CALC_FULL
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
  BasicTimeStepPtr time_steps[2] = {std::make_shared<BasicTimeStep>(options_),
                                    std::make_shared<BasicTimeStep>(options_)};
  auto lmem_allocator = std::make_shared<LmemAllocator>(options_);
  int64_t group_costs[2] = {0, 0};
  // bool pre_cost_judge = true;

  for (size_t i = 0; i < 2; ++i) {
    auto status = is_layer_group_valid(*groups[i], true, &group_costs[i]);
    if (status == false) {
      valid = false;
      break;
    }
  }
  opt_seq_info.left_cost = group_costs[0];
  opt_seq_info.right_cost = group_costs[1];
  if (!valid) {
    return false;
  }
  int64_t total_cost = group_costs[0] + group_costs[1];
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::outs() << "The final cost of the two group is " << total_cost << "\n";
  });
  if (opt_seq_info.min_cost >= 0 && opt_seq_info.min_cost <= total_cost) {
    return false;
  }
  opt_seq_info.min_cost = total_cost;

  memcpy(p_shape_secs[0], &shape_secs[0], sizeof(shape_secs_t));
  memcpy(p_shape_secs[1], &shape_secs[1], sizeof(shape_secs_t));

  return true;
}

#else

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
  BasicTimeStepPtr time_steps[2] = {std::make_shared<BasicTimeStep>(options_),
                                    std::make_shared<BasicTimeStep>(options_)};
  auto lmem_allocator = std::make_shared<LmemAllocator>(options_);
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

    std::vector<std::pair<Value, int64_t>> value_size;
    if (!init_group_data_secs(*groups[i], shape_secs[i], value_size,
                              options_)) {
      valid = false;
      break;
    }
    if (!time_steps[i]->assignTimeStep(*groups[i], shape_secs[i], true)) {
      valid = false;
      break;
    }
    if (!update_data_split(time_steps[i], *groups[i], shape_secs[i],
                           options_)) {
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
                                 time_steps[i]->get_tensor_infos(), options_)) {
        valid = false;
        break;
      }
      group_costs[i] =
          cycle_calculator_
              ->getGroupCycle(time_steps[i], shape_secs[i], groups[i]->type)
              .cycles;
    }
  }
  opt_seq_info.left_cost = group_costs[0];
  opt_seq_info.right_cost = group_costs[1];
  if (!valid) {
    return false;
  }
  int64_t total_cost = group_costs[0] + group_costs[1];
  if (pre_cost_judge) {
    LLVM_DEBUG(llvm::dbgs() << "The pre cost of the two group is " << total_cost
                            << "\n";);
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
    group_costs[i] =
        cycle_calculator_
            ->getGroupCycle(time_steps[i], shape_secs[i], groups[i]->type)
            .cycles;
  }
  opt_seq_info.left_cost = group_costs[0];
  opt_seq_info.right_cost = group_costs[1];
  if (!valid) {
    return false;
  }
  total_cost = group_costs[0] + group_costs[1];
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::outs() << "The final cost of the two group is " << total_cost << "\n";
  });
  if (opt_seq_info.min_cost >= 0 && opt_seq_info.min_cost <= total_cost) {
    return false;
  }
  opt_seq_info.min_cost = total_cost;
  memcpy(p_shape_secs[0], &shape_secs[0], sizeof(shape_secs_t));
  memcpy(p_shape_secs[1], &shape_secs[1], sizeof(shape_secs_t));

  return true;
}

#endif

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
          llvm::dbgs()
              << "; action = lg_index"
              << "; step = consider_redundant_computation_and_gdma_cost"
              << "; start_idx = " << left_cut_idx
              << "; end_idx = " << cut_result[j] << "; group_idx = " << i
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
            llvm::dbgs() << "; action = "
                         << "lg_cost"
                         << "; start_idx = " << left_cut_idx
                         << "; end_idx = " << cut_result[j]
                         << "; group_cost = " << seq_info.left_cost
                         << "; group_idx = " << i << "; step = "
                         << "consider_redundant_computation_and_gdma_cost"
                         << "; part = "
                         << "left"
                         << "\n";
            llvm::dbgs() << "; action = "
                         << "lg_cost"
                         << "; start_idx = " << cut_result[j] + 1
                         << "; end_idx = " << cut_result[j + 1]
                         << "; group_cost = " << seq_info.right_cost
                         << "; group_idx = " << i << "; step = "
                         << "consider_redundant_computation_and_gdma_cost"
                         << "; part = "
                         << "right"
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
          llvm::dbgs()
              << "; action = cut_optimize"
              << "; step = consider_redundant_computation_and_gdma_cost"
              << "; left_range = " << left_cut_idx << "-" << optimal_cut_idx
              << "; right_range = " << optimal_cut_idx + 1 << "-"
              << cut_result[j + 1] << "; group_idx = " << i << "\n";
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
                         << "; group_idx = " << i << "; action = "
                         << "lg_cost"
                         << "; step = "
                         << "merge_cut_idx_to_reduce_gdma_cost"
                         << "; part = "
                         << "left"
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
                       << "; group_idx = " << i << "; action = "
                       << "lg_cost"
                       << "; step = "
                       << "merge_cut_idx_to_reduce_gdma_cost"
                       << "; part = "
                       << "right"
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
                           << "; group_idx = " << i << "; action = "
                           << "lg_cost"
                           << "; step = "
                           << "merge_cut_idx_to_reduce_gdma_cost"
                           << "\n";
            });
            DEBUG_WITH_TYPE("cut_optimize", {
              llvm::dbgs() << "; action = cut_optimize"
                           << "; step = merge_cut_idx"
                           << "; left_range = " << start_cut_idx << "-"
                           << cut_idx << "; right_range = " << cut_idx + 1
                           << "-" << end_cut_idx << "; group_idx = " << i
                           << "\n";
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

static std::string padLeft(const std::string &s, size_t width,
                           char fill = ' ') {
  if (s.size() >= width)
    return s;
  return std::string(width - s.size(), fill) + s;
}
static std::string padRight(const std::string &s, size_t width,
                            char fill = ' ') {
  if (s.size() >= width)
    return s;
  return s + std::string(width - s.size(), fill);
}

void GroupMethod::dump_lg_idx(
    const llvm::SetVector<mlir::Operation *> &subnet_ops,
    const std::string &filename) {
  static bool initialized = false;

  std::error_code ec;
  llvm::raw_fd_ostream os(filename, ec,
                          initialized ? llvm::sys::fs::OF_Append
                                      : llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "Failed to open file: " << filename << ": " << ec.message()
                 << "\n";
    return;
  }

  constexpr size_t w_subfunc = 12;
  constexpr size_t w_op_idx = 8;

  if (!initialized) {
    os << padRight("subfunc_idx", w_subfunc) << " "
       << padRight("op_idx", w_op_idx) << " "
       << "op_ir\n";
    initialized = true;
  }

  int idx = 0;
  for (auto *op : subnet_ops) {
    std::string s_sub = std::to_string(subfunc_idx);
    std::string s_idx = std::to_string(idx++);
    os << padRight(s_sub, w_subfunc) << " " << padRight(s_idx, w_op_idx) << " ";
    op->print(os);
    os << "\n";
  }
  os.flush();
}

void GroupMethod::process(LgPassIR *pass_ir) {
  std::vector<LgInfo> &lg_infos = pass_ir->lg_infos;
  llvm::SetVector<Operation *> &subnet_ops = pass_ir->subnet_ops;
  auto start = std::chrono::high_resolution_clock::now();
  runmode_ = getRunMode(subnet_ops[0]);
  auto &lg_debugger = LgDebugger::getInstance();
  // auto func_name = pass_ir->func.getName();

  // debugger
  // 0: do nothing
  // 1: do LayerGroup and create debugger file
  // 2: only create debugger file
  // 3: do LayerGroup with debugger file
  // 4: do partial LayerGroup with debugger file
  // 5: check the single group interval given by debugger file
  std::string debugger_filename = DEBUGGER_FILE_NAME;
  if (!options_.debugger_filename.empty()) {
    debugger_filename = options_.debugger_filename;
  }
  switch (options_.debugger) {
  case 0: {
    break;
  }
  case 1: {
    lg_debugger.create_debugger_config(
        debugger_filename); // create debugger file and do LayerGroup
    break;
  }
  case 2: {
    lg_debugger.create_debugger_config(
        debugger_filename); // only create debugger file
    dump_lg_idx(subnet_ops, IDX_FILE_NAME);
    llvm::WithColor(llvm::outs(), llvm::raw_ostream::GREEN)
        << "Only create debugger file when debugger=2!\n";
    return;
  }
  case 3: // Fall through
  case 4:
  case 5:
    lg_debugger.load_debugger_config(
        debugger_filename,
        subfunc_idx); // both 3 and 4 need to load debugger file
    break;
  default: {
    llvm_unreachable("Invalid debugger option");
  }
  }
  LG_DEBUG_WITH_TYPE("lg_idx", [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "GroupMethod::process", "stamp",
                        "start processing subnet_ops with size=%d",
                        subnet_ops.size())
                 << "\n";
    llvm::dbgs() << LOG_KV("subfunc_idx", subfunc_idx) << "\n";
    int idx = 0;
    for (auto op : subnet_ops) {
      llvm::dbgs() << LOG_KV("idx", idx++)
                   << LOG_KV("op_name", module::getName(op)) << "\n";
    }
  });

  if (getenv("LOAD_TPU_GROUP") || options_.opt == 4) {
    if (is_lg_results_exists()) {
      load_lg_results(lg_infos, subnet_ops);
      if (getenv("RESEARCH_SHAPE_SECS")) {
        dump_lg_results(lg_infos);
      }
    } else {
      llvm_unreachable("file not exist's, ues opt=1/2/3 to generate");
    }
  } else if (options_.debugger == 4) {
    switch (options_.opt) {
    case 2:
      dynamic_programming_layer_group_with_cluster_debug(lg_infos, subnet_ops);
      break;
    default:
      llvm_unreachable("only opt=2 is supported when debugger=4");
      break;
    }
  } else if (options_.debugger == 5) {
    single_group_debug(lg_infos, subnet_ops);
  } else {
    switch (options_.opt) {
    case 1:
      simple_layer_group(lg_infos, subnet_ops);
      dump_lg_results(lg_infos);
      break;
    case 2:
      if (!load_hash_lg(lg_infos, subnet_ops)) {
        dynamic_programming_layer_group_with_cluster(lg_infos, subnet_ops);
        if (options_.enable_lghash)
          dump_hash_lg(lg_infos);
      }
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
  subfunc_idx += 1;
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  llvm::errs() << "GroupMethod_process time:" << elapsed.count() << "\n";
}

void GroupMethod::get_final_groups(
    std::vector<LgInfo> &lg_infos,
    const std::vector<std::vector<Operation *>> &base_groups) {
  // llvm::dbgs() << "========== get_final_groups ==========\n";
  int64_t start_idx, end_idx;
  LgInfo lg_info;
  int64_t idx_offset = 0;
  for (size_t i = 0; i < base_groups.size(); ++i) {
    start_idx = 0;
    auto &base_group = base_groups[i];
    auto &cut_result = cut_results_[i];
    for (size_t j = 0; j < cut_result.size(); ++j) {
      end_idx = cut_result[j];
      get_layer_group(lg_info, base_group, start_idx, end_idx, i, idx_offset);
      int64_t cost = -1;

      if (base_group.size() > 1) {
        if (!is_layer_group_valid(lg_info, true, &cost)) {
          llvm_unreachable("group_cost is not valid");
        }
      } else {
        if ((module::isBM1684XFamily() || module::isBM1690Family()) &&
            runmode_ == RunMode::TPU_STATIC) {
          group_one_layer_proc(lg_info, true, &cost);
          lg_info.group_cost = cost;
        }
      }
      if (lg_info.group_ops.size() >= 1 || false == options_.group_by_cores) {
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
    idx_offset += base_group.size();
  }
}

void GroupMethod::get_final_groups_by_cache(
    std::vector<LgInfo> &lg_infos,
    const std::vector<std::vector<Operation *>> &base_groups) {
  // llvm::dbgs() << "========== get_final_groups ==========\n";
  int64_t start_idx, end_idx;
  LgInfo lg_info;
  int64_t idx_offset = 0;
  for (size_t i = 0; i < base_groups.size(); ++i) {
    start_idx = 0;
    auto &base_group = base_groups[i];
    auto &cut_result = cut_results_[i];
    for (size_t j = 0; j < cut_result.size(); ++j) {
      end_idx = cut_result[j];
      get_layer_group(lg_info, base_group, start_idx, end_idx, i, idx_offset);
      int64_t cost = -1;
      // lg_info.shape_secs_search_level = 1;
      if (base_group.size() > 1 && lg_info.group_ops.size() > 1) {
        bool cache_hit;
        uint64_t hash_key;
        cache_hit =
            LgCache::getInstance().get_graph_hash(lg_info, hash_key, false);
        if (cache_hit) {
          cache_hit = LgCache::getInstance().get_info_from_cost_cache(
              hash_key, lg_info.group_cost, lg_info.shape_secs);
        }
        if (!cache_hit) {
          llvm::dbgs()
              << "[Warning]: Failed to hit cache when getting final group!";
          if (!is_layer_group_valid(lg_info, true, &cost)) {
            llvm_unreachable("group_cost is not valid");
          }
        }
      } else {
        if (module::isBM1684XFamily() && runmode_ == RunMode::TPU_STATIC) {
          group_one_layer_proc(lg_info, true, &cost);
          lg_info.group_cost = cost;
        }
      }
      if (lg_info.group_ops.size() >= 1 || false == options_.group_by_cores) {
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
    idx_offset += base_group.size();
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
      for (size_t j = 0; j < cut_result.size(); ++j) {
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
  auto filename = CACHE_FILE_NAME;
  auto ret = std::filesystem::exists(filename);
  if (!ret) {
    llvm::errs() << filename << "not exists\n";
  }
  return ret;
}

void GroupMethod::dump_lg_results(std::vector<LgInfo> &lg_infos) {
  if (!options_.lgcache) {
    return;
  }

  std::error_code EC;
  std::string filename = CACHE_FILE_NAME;

  std::filesystem::path filePath(filename);

  llvm::json::Object existingData;
  if (subfunc_idx != 0 && std::filesystem::exists(filename)) {
    auto bufferOrErr = llvm::MemoryBuffer::getFile(filename);
    if (bufferOrErr) {
      std::string content = (*bufferOrErr)->getBuffer().str();
      if (!content.empty()) {
        auto jsonOrErr = llvm::json::parse(content);
        if (jsonOrErr) {
          auto root = *jsonOrErr;
          if (auto *rootObj = root.getAsObject()) {
            for (const auto &kv : *rootObj) {
              existingData[kv.first] = kv.second;
            }
          }
        } else {
          LLVM_DEBUG(llvm::dbgs() << "Failed to parse existing JSON content: "
                                  << toString(jsonOrErr.takeError()) << "\n");
        }
      }
    }
  } else {
    existingData = llvm::json::Object{};
  }

  llvm::raw_fd_ostream OS(filename, EC);
  if (EC) {
    llvm::errs() << "Failed to open file for writing: " << EC.message() << "\n";
    return;
  }

  json::OStream J(OS, 2);
  J.objectBegin();

  J.attribute("opt", options_.opt);
  J.attribute("total subfunc", subfunc_idx + 1);

  std::vector<std::pair<int, std::string>> sortedSubfuncs;
  for (const auto &kv : existingData) {
    std::string keyStr = kv.first.str();
    if (keyStr.find("subfunc_") == 0) {
      std::string idxStr = keyStr.substr(8);
      if (!idxStr.empty()) {
        try {
          int idx = std::stoi(idxStr);
          sortedSubfuncs.emplace_back(idx, keyStr);
        } catch (const std::exception &e) {
          LLVM_DEBUG(llvm::dbgs() << "Failed to parse subfunc index from "
                                  << keyStr << ": " << e.what() << "\n");
        }
      }
    }
  }

  std::sort(sortedSubfuncs.begin(), sortedSubfuncs.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });

  for (const auto &item : sortedSubfuncs) {
    const std::string &keyStr = item.second;
    auto &subfuncData = existingData[keyStr];

    J.attributeBegin(keyStr);

    // CyclePrefixSum
    if (auto *subfuncObj = subfuncData.getAsObject()) {
      J.objectBegin();

      if (auto sfi = subfuncObj->get("subfunc idx")) {
        J.attribute("subfunc idx", sfi->getAsInteger());
      }

      if (auto groupLayer = subfuncObj->get("GroupLayer")) {
        J.attributeBegin("GroupLayer");
        J.arrayBegin();

        if (auto *groupLayerArray = groupLayer->getAsArray()) {
          for (const auto &groupObj : *groupLayerArray) {
            if (auto *groupObj_ = groupObj.getAsObject()) {
              J.objectBegin();

              if (auto base_group_idx =
                      groupObj_->getInteger("base_group_idx")) {
                J.attribute("base_group_idx", *base_group_idx);
              }
              if (auto start_idx = groupObj_->getInteger("start_idx")) {
                J.attribute("start_idx", *start_idx);
              }
              if (auto end_idx = groupObj_->getInteger("end_idx")) {
                J.attribute("end_idx", *end_idx);
              }
              if (auto func_start_idx =
                      groupObj_->getInteger("func_start_idx")) {
                J.attribute("func_start_idx", *func_start_idx);
              }
              if (auto func_end_idx = groupObj_->getInteger("func_end_idx")) {
                J.attribute("func_end_idx", *func_end_idx);
              }
              if (auto group_cost = groupObj_->getInteger("group_cost")) {
                J.attribute("group_cost", *group_cost);
              }
              if (auto index = groupObj_->getInteger("index")) {
                J.attribute("index", *index);
              }

              if (auto locsArray = groupObj_->getArray("locs")) {
                J.attributeArray("locs", [&] {
                  for (const auto &loc : *locsArray) {
                    J.value(loc);
                  }
                });
              }

              if (auto shapeSecsArray = groupObj_->getArray("shape_secs")) {
                J.attributeArray("shape_secs", [&] {
                  for (const auto &val : *shapeSecsArray) {
                    J.value(val);
                  }
                });
              }

              J.objectEnd();
            } else {
              J.value(groupObj);
            }
          }
        }

        J.arrayEnd();
        J.attributeEnd();
      }

      if (auto globalLayer = subfuncObj->get("GlobalLayer")) {
        J.attributeBegin("GlobalLayer");
        J.value(*globalLayer);
        J.attributeEnd();
      }

      if (auto cyclePrefix = subfuncObj->get("CyclePrefixSum")) {
        J.attributeBegin("CyclePrefixSum");
        J.value(*cyclePrefix);
        J.attributeEnd();
      }

      J.objectEnd();
    } else {
      J.value(subfuncData);
    }

    J.attributeEnd();
  }

  std::string subfuncKey = "subfunc_" + std::to_string(subfunc_idx);
  J.attributeBegin(subfuncKey);
  J.objectBegin();

  J.attribute("subfunc idx", subfunc_idx);

  // Write GroupLayer array
  J.attributeBegin("GroupLayer");
  J.arrayBegin();

  for (const auto &it : llvm::enumerate(lg_infos)) {
    auto group = it.value();
    if (group.group_ops.size() <= 1) {
      continue;
    }
    auto index = it.index();

    J.objectBegin();
    J.attribute("base_group_idx", group.base_group_idx);
    J.attribute("start_idx", group.start_idx);
    J.attribute("end_idx", group.end_idx);
    J.attribute("func_start_idx", group.func_start_idx);
    J.attribute("func_end_idx", group.func_end_idx);
    J.attribute("group_cost", group.group_cost);
    J.attribute("index", index); // sort_index when load

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
      J.attribute("func_idx", layer.func_end_idx);
      J.attribute("group_cost", layer.group_cost);
      J.attribute("loc", module::getName(layer.group_ops[0]).str());
      J.objectEnd();
    }
  });

  // Write cycle prefix sum
  int64_t cycle_prefix_sum = 0;
  J.attributeArray("CyclePrefixSum", [&] {
    for (const auto &it : llvm::enumerate(lg_infos)) {
      auto index = it.index();
      auto layer = it.value();
      cycle_prefix_sum += layer.group_cost;
      J.objectBegin();
      J.attribute("index", index);
      J.attribute("prefix_end_idx", layer.func_end_idx);
      J.attribute("cycle_prefix_sum", cycle_prefix_sum);
      J.objectEnd();
    }
  });

  J.objectEnd();
  J.attributeEnd();

  J.objectEnd();

  LLVM_DEBUG(llvm::dbgs() << "Dumped LG results to " << filename << "\n");
}

void GroupMethod::load_lg_results(
    std::vector<LgInfo> &lg_infos,
    const llvm::SetVector<Operation *> &subnet_ops) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(CACHE_FILE_NAME);
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
  int opt = options_.opt;
  // Load group layers

  if (auto *rootObj = root.getAsObject()) {
    if (auto opt_ = rootObj->getInteger("opt")) {
      opt = *opt_;
    } else {
      llvm_unreachable("opt not found");
    }
  }

  std::vector<std::vector<Operation *>> base_groups;
  get_base_groups(base_groups, subnet_ops);

  std::string subfuncKey = "subfunc_" + std::to_string(subfunc_idx);
  auto *subfuncData = root.getAsObject()->getObject(subfuncKey);
  if (!subfuncData) {
    llvm::errs() << "No data found for " << subfuncKey << "\n";
    return;
  }

  if (auto *subfuncData = root.getAsObject()->getObject(subfuncKey)) {

    if (auto groupLayerArray = subfuncData->getArray("GroupLayer")) {
      for (const auto &groupObj : *groupLayerArray) {
        LgInfo lg_info;

        if (auto *groupObj_ = groupObj.getAsObject()) {
          // Get operation locations
          if (auto index = groupObj_->getInteger("index")) {
            lg_info.sort_index = *index;
          }
          if (auto start_idx = groupObj_->getInteger("start_idx")) {
            lg_info.start_idx = *start_idx;
          }
          if (auto end_idx = groupObj_->getInteger("end_idx")) {
            lg_info.end_idx = *end_idx;
          }
          if (auto base_group_idx = groupObj_->getInteger("base_group_idx")) {
            lg_info.base_group_idx = *base_group_idx;
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
          } else {
            // assuming the base_group partition did not change when locs are
            // not assigned in cache
            get_layer_group(lg_info, base_groups[lg_info.base_group_idx],
                            lg_info.start_idx, lg_info.end_idx,
                            lg_info.base_group_idx);
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
    if (auto globalArray = subfuncData->getArray("GlobalLayer")) {
      for (const auto &globalObj : *globalArray) {
        LgInfo lg_info;
        if (auto *globalObj_ = globalObj.getAsObject()) {
          if (auto index = globalObj_->getInteger("index")) {
            lg_info.sort_index = *index;
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
  std::sort(lg_infos.begin(), lg_infos.end(),
            [](LgInfo &a, LgInfo &b) { return a.sort_index < b.sort_index; });

  for (auto &lg_info : lg_infos) {
    if (lg_info.group_ops.size() == 1)
      break;
    int64_t cost = 0;
    lg_info.use_cache = true;
    DEBUG_WITH_TYPE("lg_index", {
      llvm::dbgs() << "; action = lg_index"
                   << "; start_idx = " << lg_info.start_idx
                   << "; end_idx = " << lg_info.end_idx
                   << "; group_idx = " << lg_info.base_group_idx << "\n";
    });
    if (!is_layer_group_valid(lg_info, true, &cost)) {
      llvm_unreachable("group_cost is not valid");
    }
    DEBUG_WITH_TYPE("lg_cost", {
      llvm::dbgs() << "; action = lg_cost"
                   << "; step = group_layer"
                   << "; start_idx = " << lg_info.start_idx
                   << "; end_idx = " << lg_info.end_idx
                   << "; group_idx = " << lg_info.base_group_idx
                   << "; group_cost = " << lg_info.group_cost << "\n";
    });
  }
  llvm::outs() << "load lg results\n";
}

void GroupMethod::dump_hash_lg(std::vector<LgInfo> &lg_infos) {
  std::string filename = HASH_FILE_NAME;
  if (std::getenv("ENABLE_MLIR_VERSION_HASH")) {
    const char *env_value = std::getenv("ENABLE_MLIR_VERSION_HASH");
    const char *mlir_version =
        (std::strcmp(env_value, "1") == 0) ? MLIR_VERSION : env_value;
    filename = filename + "_" + mlir_version;
  }
  std::string final_path;
  std::ofstream outfile;

  std::vector<std::pair<std::string, std::string>> path_priority;

  auto env_lghash_dir = std::getenv("LGHASH_DIR");
  if (env_lghash_dir != nullptr) {
    path_priority.emplace_back(env_lghash_dir + std::string("/") + filename,
                               "environment variable LGHASH_DIR");
  }

  if (!options_.lghash_dir.empty()) {
    path_priority.emplace_back(options_.lghash_dir + "/" + filename,
                               "options.lghash_dir");
  }

  path_priority.emplace_back(filename, "current directory");

  bool file_opened = false;
  for (const auto &[path, description] : path_priority) {
    outfile.open(path, std::ios_base::app);
    if (outfile.is_open()) {
      final_path = path;
      file_opened = true;
      LLVM_DEBUG(llvm::dbgs() << "Writing hash LG to " << description << ": "
                              << path << "\n");
      break;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Failed to open " << description
                              << " path: " << path << "\n");
    }
  }

  if (!file_opened) {
    llvm::errs() << "Error: Failed to dump hash file to all available paths:\n";
    for (const auto &[path, description] : path_priority) {
      llvm::errs() << "  - " << description << ": " << path << "\n";
    }
    exit(EXIT_FAILURE);
  }

  bool fileExists = false;
  bool fileEmpty = true;
  std::ifstream infile(final_path);
  if (infile.good()) {
    fileExists = true;
    infile.seekg(0, std::ios::end);
    fileEmpty = infile.tellg() == 0;
    infile.close();
  }

  if (fileExists && !fileEmpty) {
    outfile << "\n";
  }

  outfile << "subfunc_idx: " << subfunc_idx << " ";

  for (const auto &layer : lg_infos) {
    if (layer.group_ops.size() > 1) {
      outfile << layer.func_start_idx << "," << layer.func_end_idx;
      outfile << "[" << layer.shape_secs.nsecs << "," << layer.shape_secs.csecs
              << "," << layer.shape_secs.dsecs << "," << layer.shape_secs.hsecs
              << "," << layer.shape_secs.wsecs << "]";
      outfile << ";";
    } else {
      outfile << layer.func_end_idx << ";";
    }
  }

  outfile.close();
  LLVM_DEBUG(llvm::dbgs() << "Appended hash LG to " << final_path << "\n");
}

bool GroupMethod::load_hash_lg(std::vector<LgInfo> &lg_infos,
                               const llvm::SetVector<Operation *> &subnet_ops) {
  std::string filename = HASH_FILE_NAME;
  if (std::getenv("ENABLE_MLIR_VERSION_HASH")) {
    const char *env_value = std::getenv("ENABLE_MLIR_VERSION_HASH");
    const char *mlir_version =
        (std::strcmp(env_value, "1") == 0) ? MLIR_VERSION : env_value;
    filename = filename + "_" + mlir_version;
  }
  std::ifstream infile;
  std::string final_path;
  bool file_opened = false;

  std::vector<std::pair<std::string, std::string>> path_priority;

  auto env_lghash_dir = std::getenv("LGHASH_DIR");
  if (env_lghash_dir != nullptr) {
    path_priority.emplace_back(std::string(env_lghash_dir) + "/" + filename,
                               "environment variable LGHASH_DIR");
  }

  if (!options_.lghash_dir.empty()) {
    path_priority.emplace_back(options_.lghash_dir + "/" + filename,
                               "options.lghash_dir");
  }

  path_priority.emplace_back(filename, "current directory");

  for (const auto &[path, description] : path_priority) {
    infile.open(path);
    if (infile.is_open()) {
      final_path = path;
      file_opened = true;
      LLVM_DEBUG(llvm::dbgs() << "Loading hash LG from " << description << ": "
                              << path << "\n");
      break;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Failed to open " << description
                              << " path: " << path << "\n");
    }
  }

  if (!file_opened) {
    llvm::errs() << "Failed to open hash file from all available paths:\n";
    for (const auto &[path, description] : path_priority) {
      llvm::errs() << "  - " << description << ": " << path << "\n";
    }
    return false;
  }

  lg_infos.clear();

  std::vector<Operation *> subnet_ops_vec;
  for (auto op : subnet_ops) {
    subnet_ops_vec.push_back(op);
  }

  std::string line;
  bool found = false;
  std::string target_prefix =
      "subfunc_idx: " + std::to_string(subfunc_idx) + " ";

  while (std::getline(infile, line)) {
    if (line.rfind(target_prefix, 0) == 0) {
      found = true;
      break;
    }
  }

  if (!found) {
    llvm::errs() << "No data found for subfunc_idx: " << subfunc_idx << "\n";
    infile.close();
    return false;
  }

  infile.close();

  std::string data = line.substr(target_prefix.length());
  std::istringstream iss(data);
  std::string token;
  int index = 0;

  // auto &lg_config = LgConfig::getInstance();

  while (std::getline(iss, token, ';')) {
    if (token.empty())
      continue;

    LgInfo lg_info;
    lg_info.sort_index = index++;

    size_t bracket_start = token.find('[');
    size_t bracket_end = token.find(']');

    std::string indices_str = token;
    std::string shapes_str = "";

    if (bracket_start != std::string::npos &&
        bracket_end != std::string::npos && bracket_end > bracket_start) {
      indices_str = token.substr(0, bracket_start);
      shapes_str =
          token.substr(bracket_start + 1, bracket_end - bracket_start - 1);
    }

    size_t comma_pos = indices_str.find(',');
    if (comma_pos != std::string::npos) {
      std::string start_str = indices_str.substr(0, comma_pos);
      std::string end_str = indices_str.substr(comma_pos + 1);

      if (start_str.empty() || end_str.empty()) {
        llvm::errs() << "Empty integer string: " << indices_str << "\n";
        continue;
      }

      std::istringstream start_ss(start_str);
      std::istringstream end_ss(end_str);
      int start_idx, end_idx;

      if (!(start_ss >> start_idx) || !(end_ss >> end_idx)) {
        llvm::errs() << "Invalid integer format: " << indices_str << "\n";
        continue;
      }

      if (!start_ss.eof() || !end_ss.eof()) {
        llvm::errs() << "Invalid integer format (extra characters): "
                     << indices_str << "\n";
        continue;
      }

      get_layer_group(lg_info, subnet_ops_vec, start_idx, end_idx, -1);
      lg_info.func_start_idx = start_idx;
      lg_info.func_end_idx = end_idx;
    } else {
      if (indices_str.empty()) {
        llvm::errs() << "Empty integer string: " << indices_str << "\n";
        continue;
      }

      std::istringstream idx_ss(indices_str);
      int idx;

      if (!(idx_ss >> idx)) {
        llvm::errs() << "Invalid integer format: " << indices_str << "\n";
        continue;
      }

      if (!idx_ss.eof()) {
        llvm::errs() << "Invalid integer format (extra characters): "
                     << indices_str << "\n";
        continue;
      }

      if (idx >= 0 && static_cast<size_t>(idx) < subnet_ops_vec.size()) {
        lg_info.group_ops.push_back(subnet_ops_vec[idx]);
        lg_info.func_start_idx = idx;
        lg_info.func_end_idx = idx;
      } else {
        llvm::errs() << "Index out of range: " << idx << "\n";
        continue;
      }
    }

    if (!shapes_str.empty()) {
      std::istringstream shape_ss(shapes_str);
      std::string dim_str;
      std::vector<int> dims;

      while (std::getline(shape_ss, dim_str, ',')) {
        if (!dim_str.empty()) {
          try {
            dims.push_back(std::stoi(dim_str));
          } catch (const std::exception &e) {
            llvm::errs() << "Invalid shape dimension: " << dim_str << "\n";
            dims.clear();
            break;
          }
        }
      }

      if (dims.size() == 5) {
        lg_info.shape_secs.nsecs = dims[0];
        lg_info.shape_secs.csecs = dims[1];
        lg_info.shape_secs.dsecs = dims[2];
        lg_info.shape_secs.hsecs = dims[3];
        lg_info.shape_secs.wsecs = dims[4];
        lg_info.is_best_shape_secs = true;
      } else if (!dims.empty()) {
        llvm::errs()
            << "Invalid shape_secs format, expected 5 dimensions, got: "
            << dims.size() << "\n";
      }
    }

    set_group_type(lg_info);
    lg_info.update_group_io(options_.opt);

    if (lg_info.group_ops.size() > 1) {
      bool status;
      auto time_step = std::make_shared<BasicTimeStep>(options_);
      status = time_step->assignTimeStep(lg_info, lg_info.shape_secs, true);
      if (!status) {
        llvm::errs() << "load lghash time_step error"
                     << "\n";
        lg_infos.clear();
        return false;
      }
      auto lmem_allocator = std::make_shared<LmemAllocator>(options_);
      status = lmem_allocator->assignLmemAddrWithSecs(
          lg_info, time_step, lg_info.shape_secs, false, true);
      if (!status) {
        llvm::errs() << "load lghash SHAPESECS error"
                     << "\n";
        lg_infos.clear();
        return false;
      }
      // uint64_t hash_key;
      // LgCostCache::getInstance().cache_enabled = true;
      // LgCostCache::getInstance().get_graph_hash(lg_info, hash_key);
      // LgCostCache::getInstance().add_cache(hash_key, lg_info);
    }

    // if (lg_config.get_shape_secs_search_strategy() ==
    //     SHAPE_SECS_ALWAYS_BETTER) {
    //   lg_info.shape_secs_search_level = 1;
    // }

    // int64_t group_cost = 0;
    // if (lg_info.group_ops.size() > 1 &&
    //     !is_layer_group_valid(lg_info, true, &group_cost)) {
    //   llvm::errs() << "Invalid layer group detected\n";
    //   lg_infos.clear();
    //   return false;
    // }

    lg_infos.push_back(lg_info);
  }

  std::sort(lg_infos.begin(), lg_infos.end(),
            [](LgInfo &a, LgInfo &b) { return a.sort_index < b.sort_index; });

  llvm::outs() << "Loaded hash LG from " << final_path << "\n";
  return true;
}
/// The pass of layer group searching
class LayerGroupSearchPass : public LgPass {
public:
  LayerGroupSearchPass(const LgOptions &options) { options_ = options; }
  virtual bool run(LgPassIR *pass_ir) override {
    auto group_method = GroupMethod(options_);
    group_method.process(pass_ir);
    return true;
  }
  virtual std::string name() override { return "LayerGroupSearchPass"; }
  virtual std::string brief() override {
    return "Searching the optimal layer groups";
  }
};

std::unique_ptr<LgPass> CreateLayerGroupSearchPass(const LgOptions &options) {
  return std::unique_ptr<LgPass>(new LayerGroupSearchPass(options));
}

} // namespace tpu
} // namespace tpu_mlir

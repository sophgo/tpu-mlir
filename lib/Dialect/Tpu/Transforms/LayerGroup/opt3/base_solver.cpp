
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "progressbar.hpp"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Backend/BM168x/BackendInterfaces.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupMethod.h"
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

bool opt_cost_all = false;

bool Sort_by_int(const std::pair<Value, int64_t> &v1,
                 const std::pair<Value, int64_t> &v2) {
  return v1.second < v2.second;
}

bool SortByCycle(const ts_cycle_info &v1, const ts_cycle_info &v2) {
  return v1.cycle > v2.cycle; //降序排列
}
bool SortByCycleDiff(const ts_cycle_info &v1, const ts_cycle_info &v2) {
  return v1.cycle_diff > v2.cycle_diff; //降序排列
}

bool pair_op_int_Sort_by_int(const std::pair<Operation *, int> &v1,
                             const std::pair<Operation *, int> &v2) {
  return v1.second < v2.second;
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

std::string GenerateRandomString(int length) {
  std::string charset =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, charset.length() - 1);

  std::string s;
  s.reserve(length);
  for (int i = 0; i < length; ++i) {
    s += charset[dis(gen)];
  }
  return s;
}

class ilp_func_trace {
public:
  ilp_func_trace(std::string debug_info, int64_t specified_id = -1,
                 std::shared_ptr<dot_graph> dot_graph_log = nullptr) {
    _debug_info = debug_info;
    _dot_graph_log = dot_graph_log;
    string_id = specified_id == -1 ? GenerateRandomString(15)
                                   : std::to_string(specified_id);
    LAYER_GROUP_LOG_DEBUG_BLOCK({
      llvm::errs() << string_id << " ilp_debug: " << _debug_info << " start\n";
    });
  }

  ~ilp_func_trace() {
    std::string extra_info = "";
    if (_dot_graph_log) {
      std::string svg_file =
          _dot_graph_log->export_dot("svg_" + _debug_info + "_" + string_id);
      extra_info = ", please refer svg:" + svg_file;
    }
    LAYER_GROUP_LOG_DEBUG_BLOCK({
      llvm::errs() << string_id << " ilp_debug: " << _debug_info << " end"
                   << extra_info << "\n";
    });
    if (call_exit) {
      exit(1);
    }
  }

  void need_exit() { call_exit = true; }

private:
  bool call_exit = false;
  std::string string_id;
  std::string _debug_info;
  std::shared_ptr<dot_graph> _dot_graph_log;
};

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
  if (module::isMARS3() || module::isSGTPUV8())
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

static void topo_order_dfs(Operation *cur_op, std::vector<Operation *> ops,
                           std::map<Operation *, int> &indeg,
                           std::vector<Operation *> &topo_ops) {
  topo_ops.push_back(cur_op);
  for (auto user : cur_op->getUsers()) {
    if (std::find(ops.begin(), ops.end(), user) != ops.end()) {
      indeg[user] = indeg[user] - 1;
      if (indeg[user] == 0) {
        if (std::find(topo_ops.begin(), topo_ops.end(), user) ==
            topo_ops.end()) {
          topo_order_dfs(user, ops, indeg, topo_ops);
        }
      }
    }
  }
}

static void find_op_tree_by_root(Operation *op,
                                 std::vector<Operation *> &op_tree,
                                 std::vector<Operation *> ops) {
  op_tree.push_back(op);
  for (auto user : op->getUsers()) {
    if (std::find(ops.begin(), ops.end(), user) != ops.end()) {
      find_op_tree_by_root(user, op_tree, ops);
    }
  }
}

void GroupMethod::get_base_dfs_topo_groups(
    std::vector<std::shared_ptr<ilp_LgInfo>> &tmp_base_groups) {
  int idx = 0;
  for (auto &grp : tmp_base_groups) {
    auto &ops = grp->_lgInfo.group_ops;
    idx++;
    if (ops.size() == 1) {
      continue;
    }
    llvm::errs() << "start refine order, grp:" << --idx << "\n";
    std::vector<Operation *> topo_ops;
    std::map<Operation *, int> indeg;
    for (auto op : ops) {
      indeg[op] = 0;
      for (auto v : op->getOperands()) {
        if (std::find(ops.begin(), ops.end(), v.getDefiningOp()) != ops.end()) {
          if (indeg.find(op) != indeg.end()) {
            indeg[op] = indeg[op] + 1;
          }
        }
      }
    }
    for (auto it : indeg) {
      if (it.second == 0) {
        if (std::find(topo_ops.begin(), topo_ops.end(), it.first) ==
            topo_ops.end()) {
          topo_order_dfs(it.first, ops, indeg, topo_ops);
        }
      }
    }

    int i = 0;
    llvm::errs() << "full_topo_ops:\n";
    for (auto op : topo_ops) {
      llvm::errs() << "  op:" << i++ << ": " << show_op_info(op) << "\n";
    }
    ops.assign(topo_ops.begin(), topo_ops.end());
  }
}

static inline int64_t increase_nsecs2(int64_t nsecs, int64_t max_nsecs) {
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

  return next_nsecs;
}

static inline int64_t increase_csecs2(int64_t csecs, int64_t max_csecs) {
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

  return next_csecs;
}

static inline bool
update_shape_secs_for_ilp_group(shape_secs_t &shape_secs,
                                const shape_secs_t &max_shape_secs) {
  bool updated = false;
  if (shape_secs.nsecs < max_shape_secs.nsecs) {
    shape_secs.nsecs++;
    updated = true;
  } else if (shape_secs.hsecs < max_shape_secs.hsecs) {
    shape_secs.hsecs++;
    updated = true;
  } else if (shape_secs.csecs < max_shape_secs.csecs) {
    shape_secs.csecs++;
    updated = true;
  }

  return updated;
}

void get_sec_per_cores(
    ilp_LgInfo &sub_group,
    std::map<int64_t, std::vector<std::vector<int64_t>>> &vec_ncdhw,
    int core_num, slice_info_t &slice_info) {
  bool only_nc = sub_group.p_special_grp
                     ? sub_group.p_special_grp->name() == "attention_group" &&
                           !sub_group.p_special_grp->hdim_is_batch
                     : false;
  auto shape_secs = sub_group.shape_secs;
  int secs = shape_secs.get_sec_num(only_nc);
  if (secs < core_num) {
    only_nc = false;
    secs = shape_secs.get_sec_num(only_nc);
  }
  int secs_per_core = secs / core_num;
  vec_ncdhw.clear();
  std::vector<std::vector<int64_t>> tmp_ncdhws;
  if (shape_secs.n_slice_num > 0) {
    if (sub_group.p_special_grp->hdim_is_batch) {
      int n_slice = shape_secs.n_slice_num, h_slice = 1;
      if (shape_secs.n_slice_num > shape_secs.shape_0) {
        n_slice = shape_secs.shape_0;
        h_slice = shape_secs.n_slice_num / shape_secs.shape_0;
      }
      for (int h = 0; h < h_slice; h++) {
        for (int c = 0; c < shape_secs.c_slice_num; c++) {
          for (int n = 0; n < n_slice; n++) {
            std::vector<int64_t> tmp = {n, c, 0, h, 0};
            tmp_ncdhws.push_back(tmp);
          }
        }
      }
    } else {
      int h_slice_num = only_nc ? 1 : shape_secs.h_slice_num;
      for (int h = 0; h < h_slice_num; h++) {
        for (int c = 0; c < shape_secs.c_slice_num; c++) {
          for (int n = 0; n < shape_secs.n_slice_num; n++) {
            std::vector<int64_t> tmp = {n, c, 0, h, 0};
            tmp_ncdhws.push_back(tmp);
          }
        }
      }
    }
  } else {
    for (int n = 0; n < shape_secs.nsecs; n++) {
      for (int c = 0; c < shape_secs.csecs; c++) {
        for (int d = 0; d < shape_secs.dsecs; d++) {
          for (int h = 0; h < shape_secs.hsecs; h++) {
            for (int w = 0; w < shape_secs.wsecs; w++) {
              std::vector<int64_t> tmp = {n, c, d, h, w};
              tmp_ncdhws.push_back(tmp);
            }
          }
        }
      }
    }
  }

  int idx = 0;
  for (int i = 0; i < core_num; i++) {
    vec_ncdhw[i] = std::vector<std::vector<int64_t>>();
    for (int j = 0; j < secs_per_core; j++) {
      vec_ncdhw[i].push_back(tmp_ncdhws[idx++]);
    }
  }
  int rest = secs - core_num * secs_per_core;
  if (rest > 0) {
    for (int i = 0; i < core_num; i++) {
      if (--rest < 0) {
        break;
      }
      vec_ncdhw[i].push_back(tmp_ncdhws[idx++]);
    }
  }

  if (module::isDebugCmdEnable("detail_info_show")) {
    llvm::errs() << "vec_ncdhw, core num:" << vec_ncdhw.size() << "\n";
    for (int i = 0; i < vec_ncdhw[0].size(); i++) {
      llvm::errs() << "  row" << i << "\n";
      for (int j = 0; j < core_num; j++) {
        if (i < vec_ncdhw[j].size()) {
          auto &ncdhw = vec_ncdhw[j][i];
          llvm::errs() << "    col" << j << "[" << ncdhw[0] << "," << ncdhw[1]
                       << "," << ncdhw[2] << "," << ncdhw[3] << "," << ncdhw[4]
                       << "]  ";
          llvm::errs() << "slice:[" << slice_info.n[ncdhw[0]].second << ","
                       << slice_info.c[ncdhw[1]].second << ","
                       << slice_info.d[ncdhw[2]].second << ","
                       << slice_info.h[ncdhw[3]].second << ","
                       << slice_info.w[ncdhw[4]].second << "]";
        }
      }
      llvm::errs() << "\n";
    }
  }
}

// std::vector<std::pair<int,int>> get_var_low_high_bound(int slice_num, int
// group_size, int overlap_size = 1) {
//   std::vector<std::pair<int,int>> tmp;
//   int end = slice_num*group_size + 1;
//   if (slice_num == 1) {
//     tmp.push_back(std::make_pair(0, end));
//   } else if (slice_num == 2) {
//     tmp.push_back(std::make_pair(0, group_size + 1 + overlap_size));
//     tmp.push_back(std::make_pair(group_size - overlap_size + 1, end));
//   } else {
//     tmp.push_back(std::make_pair(0, group_size + 1 + overlap_size));
//     for (int i = 0; i < slice_num - 2) {
//       tmp.push_back(std::make_pair(0, group_size + 1 + overlap_size));
//     }
//     tmp.push_back(std::make_pair(0, 2*group_size + 1));
//   }
//   return std::move(tmp);
// }

// std::vector<std::pair<int,int>> get_var_low_high_bound(int slice_num, int
// group_size, int overlap_size = 1) {
//   std::vector<std::pair<int,int>> tmp;
//   int end = slice_num*group_size + 1;
//   if (slice_num == 1) {
//     tmp.push_back(std::make_pair(0, end));
//   } else if (slice_num == 2) {
//     tmp.push_back(std::make_pair(0, group_size + 1 + overlap_size));
//     tmp.push_back(std::make_pair(group_size - overlap_size + 1, end));
//   } else {
//     tmp.push_back(std::make_pair(0, group_size + 1 + overlap_size));
//     for (int i = 0; i < slice_num - 2) {
//       tmp.push_back(std::make_pair(0, group_size + 1 + overlap_size));
//     }
//     tmp.push_back(std::make_pair(0, 2*group_size + 1));
//   }
//   return std::move(tmp);
// }

std::vector<op_var_pos_info> createOverlapStrategy(const LgInfo &lg_info,
                                                   int slice_num, int type = 0,
                                                   int overlap = 2,
                                                   int fix_gap = 4) {
  std::vector<op_var_pos_info> op_var_bound;
  op_var_pos_info null_var_pos;
  null_var_pos.ts_id = 0;
  op_var_bound.push_back(null_var_pos);
  int k = 1;
  int op_num = lg_info.group_ops.size();
  LAYER_GROUP_LOG_DEBUG_BLOCK(
      { llvm::errs() << "old overlap:" << overlap << "\n"; });
  if (op_num <= overlap) {
    overlap = 1;
  } else if (op_num * 0.2 > overlap) {
    overlap = op_num * 0.2;
  }
  LAYER_GROUP_LOG_DEBUG_BLOCK(
      { llvm::errs() << "new overlap:" << overlap << "\n"; });
  for (int n = 0; n < slice_num; n++) {
    int group_offset = k;
    for (int m = 0; m < op_num; m++) {
      op_var_pos_info var_pos;
      var_pos.ts_id = k++;
      var_pos.key = std::make_pair(n, m);
      if (type == 0) {
        var_pos.start_ts = group_offset - overlap;
        var_pos.end_ts = group_offset + op_num + overlap - 1;
      } else if (type == 1) {
        var_pos.start_ts = var_pos.ts_id - fix_gap;
        var_pos.end_ts = var_pos.ts_id + fix_gap;
      }

      if (var_pos.start_ts < 0) {
        var_pos.start_ts = 0;
      }
      if (var_pos.end_ts >= slice_num * op_num + 2) {
        var_pos.end_ts = slice_num * op_num + 1;
      }
      op_var_bound.push_back(var_pos);
    }

    // if (type == 1 && n != slice_num - 1) {
    //   null_var_pos.ts_id = k++;
    //   op_var_bound.push_back(null_var_pos);
    // }
  }
  null_var_pos.ts_id = k;
  op_var_bound.push_back(null_var_pos);
  return std::move(op_var_bound);
}

void showTensorInfo(TensorInfo tensor_infos) {
  LOG(INFO) << "showTensorInfo:";
  for (auto itr = tensor_infos.begin(); itr != tensor_infos.end(); ++itr) {
    LOG(INFO) << " tensor name: " << module::getName(itr->first).str();
    int i = 0;
    for (auto itr2 : itr->second.slice_info.n) {
      LOG(INFO) << "  n: " << i << " (" << itr2.first << "," << itr2.second
                << ")";
      i++;
    }
    i = 0;
    for (auto itr2 : itr->second.slice_info.h) {
      LOG(INFO) << "  h: " << i << " (" << itr2.first << "," << itr2.second
                << ")";
      i++;
    }
  }
}

static bool isDifferenceWithinFivePercent(int64_t new_value,
                                          int64_t old_value) {
  if (new_value < old_value) {
    return true;
  }
  int64_t difference = std::abs(new_value - old_value);
  int64_t maxValue = std::max(new_value, old_value);
  double fivePercent = 0.05 * maxValue;
  return difference <= fivePercent;
}

void GroupMethod::cut_this_group_is_better(
    ilp_LgInfo &original_group, LgPassIR *pass_ir,
    std::vector<std::shared_ptr<ilp_LgInfo>> &base_groups,
    std::vector<Operation *> &processed_ops, bool is_cut_op_is_global) {
  std::vector<std::pair<Operation *, ts_cycle_info>> cut_ops;
  auto ilp_timeStep = original_group.timeStepPtrs[0];
  int group_cycle, group_cycle_diff;
  std::vector<ts_cycle_info> ts_cycle;
  ilp_timeStep->get_group_cycle_info(group_cycle, group_cycle_diff, ts_cycle);
  auto group_ops = original_group._lgInfo.group_ops;
  if (group_ops.size() > 3) {
    std::vector<ts_cycle_info> ts_cycle2;
    ts_cycle2.assign(ts_cycle.begin(), ts_cycle.end());
    std::sort(ts_cycle2.begin(), ts_cycle2.end(), SortByCycleDiff);
    std::sort(ts_cycle.begin(), ts_cycle.end(), SortByCycle);
    int i = 0;
    if (module::isDebugCmdEnable("detail_info_show")) {
      for (auto [it1, it2] : llvm::zip(ts_cycle, ts_cycle2)) {
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::errs() << "separate_grp, i:" << i++
                       << ", ts_cycle, ts_idx:" << it1.ts_idx
                       << ", cycle:" << it1.cycle
                       << ", cycle_diff:" << it1.cycle_diff << ", "
                       << show_op_info(it1.cut_op)
                       << ", ts_cycle2, ts_idx:" << it2.ts_idx
                       << ", cycle:" << it2.cycle
                       << ", cycle_diff:" << it2.cycle_diff << ", "
                       << show_op_info(it2.cut_op) << "\n";
        });
      }
    }

    int top_diff_num = 6;
    for (int i = 0; i < top_diff_num; i++) {
      float ratio = ts_cycle[i].cycle_diff;
      ratio /= ts_cycle[i].cycle;
      if (ts_cycle[i].ts_idx > 0 && ratio > 0.5) {
        for (int j = 0; j < top_diff_num; j++) {
          if (ts_cycle[i].ts_idx == ts_cycle2[j].ts_idx &&
              ts_cycle2[j].ts_idx != 0) {
            auto cut_op = ts_cycle[i].cut_op;
            if (is_cut_op_is_global) {
              if (cut_op && isa<tpu::Conv2DOp>(cut_op)) {
                LAYER_GROUP_LOG_DEBUG_BLOCK({
                  llvm::errs()
                      << "separate_grp, is_cut_op_is_global, split group at "
                      << show_op_info(cut_op) << "\n";
                });
                cut_ops.push_back(std::make_pair(cut_op, ts_cycle[i]));
              }
            } else {
              if (cut_op) {
                LAYER_GROUP_LOG_DEBUG_BLOCK({
                  llvm::errs() << "separate_grp, split group at "
                               << show_op_info(cut_op) << "\n";
                });
                cut_ops.push_back(std::make_pair(cut_op, ts_cycle[i]));
              }
            }
          }
        }
      }
    }
  }

  if (cut_ops.size() == 0) {
    LAYER_GROUP_LOG_DEBUG_BLOCK(
        { llvm::errs() << "separate_grp, cut_ops empty\n"; });
    original_group.conv_cut_optimized = true;
    return;
  }

  for (auto it : cut_ops) {
    auto cut_op = it.first;
    if (opHasMultiGroupUser(cut_op, group_ops)) {
      continue;
    }
    std::vector<Operation *> left_sub_ops;
    std::vector<std::shared_ptr<ilp_LgInfo>> tmp_groups;
    left_sub_ops.assign(group_ops.begin(), group_ops.end());
    std::vector<Operation *> right_sub_ops;
    find_all_next_ops(cut_op, right_sub_ops, &group_ops);
    for (auto right_sub_op : right_sub_ops) {
      left_sub_ops.erase(
          std::remove(left_sub_ops.begin(), left_sub_ops.end(), right_sub_op),
          left_sub_ops.end());
    }
    if (is_cut_op_is_global) {
      right_sub_ops.erase(
          std::remove(right_sub_ops.begin(), right_sub_ops.end(), cut_op),
          right_sub_ops.end());
    } else {
      if (it.second.mode == 2 ||
          (it.second.mode == 0 && it.second.load_cycle_is_bigger)) {
        LAYER_GROUP_LOG_DEBUG_BLOCK(
            { llvm::errs() << "separate_grp, cut_op in left_sub_ops\n"; });
        right_sub_ops.erase(
            std::remove(right_sub_ops.begin(), right_sub_ops.end(), cut_op),
            right_sub_ops.end());
        left_sub_ops.push_back(cut_op);
      }
    }

    uint64_t left_sub_group_cost = std::numeric_limits<uint64_t>::max();
    uint64_t right_sub_group_cost = std::numeric_limits<uint64_t>::max();
    if (left_sub_ops.size() > 0) {
      if (left_sub_ops.size() == 1) {
        left_sub_group_cost =
            cycle_calculator_->getGlobalLayerCycle(left_sub_ops.back());
      } else {
        auto left_sub_group =
            CreateIlpLgInfo(sortOpsByOtherOpsOrder(group_ops, left_sub_ops),
                            options_, STRATEGY_SEARCH_CONV_CUT);
        if (original_group.p_special_grp) {
          left_sub_group->p_special_grp = original_group.p_special_grp;
          if (!original_group.p_special_grp->convert_to_other_type(
                  left_sub_ops, left_sub_group->p_special_grp)) {
            LAYER_GROUP_LOG_DEBUG_BLOCK({
              llvm::errs() << "separate_grp: matmul grp convert_to_other_type "
                              "fail in cut_this_group_is_better\n";
            });
            left_sub_group->p_special_grp = nullptr;
            // continue;
          } else {
            left_sub_group->_lgInfo.type = GROUP_MM_OPT3;
          }
        }
        left_sub_group->base_solver(pass_ir, cycle_calculator_);
        if (left_sub_group->group_cycle > 0) { // Make sure the group succeeds
          left_sub_group_cost = left_sub_group->group_cycle;
          left_sub_group->group_success = true;
          tmp_groups.push_back(left_sub_group);
        } else {
          continue;
        }
      }
    } else {
      continue;
    }

    if (right_sub_ops.size() > 0) {
      if (right_sub_ops.size() == 1) {
        right_sub_group_cost =
            cycle_calculator_->getGlobalLayerCycle(right_sub_ops.back());
      } else {
        auto right_sub_group =
            CreateIlpLgInfo(sortOpsByOtherOpsOrder(group_ops, right_sub_ops),
                            options_, STRATEGY_SEARCH_CONV_CUT);
        if (original_group.p_special_grp) {
          right_sub_group->p_special_grp = original_group.p_special_grp;
          if (!original_group.p_special_grp->convert_to_other_type(
                  left_sub_ops, right_sub_group->p_special_grp)) {
            LAYER_GROUP_LOG_DEBUG_BLOCK({
              llvm::errs() << "separate_grp: matmul grp convert_to_other_type "
                              "fail in cut_this_group_is_better\n";
            });
            right_sub_group->p_special_grp = nullptr;
            // continue;
          } else {
            right_sub_group->_lgInfo.type = GROUP_MM_OPT3;
          }
        }
        right_sub_group->base_solver(pass_ir, cycle_calculator_);
        if (right_sub_group->group_cycle > 0) {
          right_sub_group_cost = right_sub_group->group_cycle;
          right_sub_group->group_success = true;
          tmp_groups.push_back(right_sub_group);
        } else {
          continue;
        }
      }
    } else {
      continue;
    }

    int64_t global_op_cost =
        is_cut_op_is_global ? cycle_calculator_->getGlobalLayerCycle(cut_op)
                            : 0;
    int64_t cut_group_cost =
        left_sub_group_cost + right_sub_group_cost + global_op_cost;
    LAYER_GROUP_LOG_DEBUG_BLOCK({
      llvm::errs() << "separate_grp, left_sub_group_cost:"
                   << left_sub_group_cost
                   << ", right_sub_group_cost:" << right_sub_group_cost
                   << ", global_op_cost:" << global_op_cost
                   << ", cut_group_cost:" << cut_group_cost
                   << ", original_group_cycle:" << original_group.group_cycle
                   << ", for " << show_op_info(cut_op) << "\n";
    });
    if (isDifferenceWithinFivePercent(cut_group_cost,
                                      original_group.group_cycle)) {
      processed_ops.push_back(cut_op);
      base_groups.insert(base_groups.end(), tmp_groups.begin(),
                         tmp_groups.end());
      for (auto [index, tmp_group] : llvm::enumerate(tmp_groups)) {
        LAYER_GROUP_LOG_DEBUG_BLOCK(
            { llvm::errs() << ">>>tmp_group" << index << ":\n"; });
        tmp_group->_lgInfo.dump_lginfo();
      }
      LAYER_GROUP_LOG_DEBUG_BLOCK({
        llvm::errs() << "separate_grp, add sub group for "
                     << show_op_info(cut_op) << "\n";
      });
      original_group.group_success = false;
      break;
    } else {
      LAYER_GROUP_LOG_DEBUG_BLOCK(
          { llvm::errs() << "separate_grp, invalid\n"; });
    }
  }
  original_group.conv_cut_optimized = true;
  return;
}

void GroupMethod::try_modify_mlp_group_sub_sum(
    LgPassIR *pass_ir, std::vector<std::shared_ptr<ilp_LgInfo>> &base_groups) {
  std::map<Value, Value, value_compare> map_old_v_to_new_v;
  if (module::isDebugCmdEnable("disable_mlp_sub_sum")) {
    return;
  }
  for (int group_idx = 0; group_idx < base_groups.size(); group_idx++) {
    auto grp = base_groups[group_idx];
    bool input_changed = false;

    // this is for the case that the mlp group_out is input of other groups
    // first replace _lgInfo.group_ins according to map which store before
    // then modify the createLoadOp2 func in build_mlir.cpp
    for (auto &opd : grp->_lgInfo.group_ins) {
      if (map_old_v_to_new_v.find(opd) == map_old_v_to_new_v.end()) {
        continue;
      }
      auto &map_old_v_to_new_v_from_lgInfo =
          pass_ir->map_old_v_to_new_v_in_group_in;
      map_old_v_to_new_v_from_lgInfo[opd] = map_old_v_to_new_v[opd];
      auto old_opd = opd;
      opd = map_old_v_to_new_v[old_opd];
      input_changed = true;
      // auto& ILP_time_steps = grp->timeStepPtrs[0];
      // for (int ts = 0; ts < ILP_time_steps->timestep_table_new.size(); ts++){
      //   for (auto& it: ILP_time_steps->timestep_table_new[ts].vec_ts_var){
      //     if (it.value == old_opd){
      //       it.value = opd;
      //     }
      //   }
      // }
    }

    if (!grp->group_success || grp->p_special_grp == nullptr ||
        grp->p_special_grp->name() != "mlp_group" ||
        grp->shape_secs.h_slice_num == 1) {
      continue;
    }

    auto &_lgInfo = grp->_lgInfo;
    Operation *second_matmul_op;
    if (_lgInfo.type == GROUP_MM_OPT3) {
      int cnt_mlp = 0;
      for (auto op : _lgInfo.group_ops) {
        if (auto _second_matmul_op = dyn_cast_or_null<tpu::MatMulOp>(op)) {
          cnt_mlp++;
          if (cnt_mlp == 2) {
            second_matmul_op = _second_matmul_op;
          }
        }
      }
      auto grp_out = second_matmul_op->getResults()[0];
      auto builder = OpBuilder(grp_out.getUsers().begin()->getContext());
      auto do_bias =
          std::find(_lgInfo.group_ins.begin(), _lgInfo.group_ins.end(),
                    second_matmul_op->getOperands()[2]) !=
          _lgInfo.group_ins.end();
      auto do_relu =
          second_matmul_op->getAttr("do_relu").cast<BoolAttr>().getValue();
      if (!do_relu && !do_bias) {
        continue;
      }

      second_matmul_op->setAttr("do_relu", builder.getBoolAttr(false));
      if (!do_bias) {
        builder.setInsertionPointAfterValue(grp_out);
        auto loc = module::getLocLike(grp_out, "addconst_for_mlp_part_sum");
        auto add = builder.create<tpu::AddConstOp>(loc, grp_out.getType(),
                                                   mlir::ValueRange{grp_out});
        add->setAttr("do_relu", builder.getBoolAttr(do_relu));
        add->setAttr("const_val", builder.getF64FloatAttr(0.0));
        grp_out.replaceAllUsesExcept(add->getResults()[0], add);
      } else {
        // cancle bias
        auto bias_op = second_matmul_op->getOperands()[2];
        auto None_op = second_matmul_op->getOperands()[3];
        second_matmul_op->setOperand(2, None_op);

        // add reshapeOp
        auto loc =
            module::getLocLike(grp_out, "add_for_mlp_part_sum_bias_reshape");
        builder.setInsertionPointAfterValue(grp_out);
        auto shape_size = module::getShape(grp_out).size();
        std::vector<int64_t> new_shape(shape_size - 1, 1);
        new_shape.push_back(module::getShape(grp_out)[shape_size - 1]);
        std::vector<NamedAttribute> attrs;
        attrs.emplace_back(
            builder.getNamedAttr("shape", builder.getI64ArrayAttr(new_shape)));
        auto reshape = builder.create<tpu::ReshapeOp>(
            loc,
            RankedTensorType::get(new_shape, module::getElementType(grp_out)),
            mlir::ValueRange{bias_op}, attrs);

        // add addOp
        loc = module::getLocLike(grp_out, "add_for_mlp_part_sum");
        builder.setInsertionPointAfter(reshape);
        auto add = builder.create<tpu::AddOp>(
            loc, grp_out.getType(),
            mlir::ValueRange{grp_out, reshape->getResults()[0]});
        add->setAttr("do_relu", builder.getBoolAttr(do_relu));
        grp_out.replaceAllUsesExcept(add->getResults()[0], add);
        map_old_v_to_new_v[grp_out] = add->getResults()[0];

        // cast if needed
        if (bias_op.getType().isF16() && grp_out.getType().isF32()) {
          loc = module::getLocLike(grp_out, "add_for_mlp_part_sum_bias_cast");
          builder.setInsertionPointAfterValue(bias_op);
          std::vector<int64_t> weight_shape{1, 1, 1,
                                            module::getShape(bias_op)[3]};
          auto cast = builder.create<tpu::CastOp>(
              loc, RankedTensorType::get(weight_shape, builder.getF16Type()),
              mlir::ValueRange{bias_op}, attrs);
          reshape.setOperand(0, cast->getResults()[0]);
        }
      }
    }
  }
}

static bool need_exit(std::vector<Operation *> processed_ops) {
  std::map<Operation *, int> countMap;
  for (auto op : processed_ops) {
    countMap[op]++;
  }
  int max_count = 0;
  for (auto it : countMap) {
    if (it.second > max_count) {
      max_count = it.second;
    }
  }
  return max_count > 10;
}

void GroupMethod::try_cut_some_group(
    LgPassIR *pass_ir, std::vector<std::shared_ptr<ilp_LgInfo>> &base_groups,
    bool is_cut_op_is_global) {
  if (module::isDebugCmdEnable("disable_group_cut")) {
    return;
  }
  ilp_func_trace tmp_trace(__func__);
  for (auto &grp : base_groups) {
    grp->conv_cut_optimized = false;
    grp->group_success = true;
  }
  std::vector<Operation *> processed_ops;
  while (true) {
    bool all_optimized = true;
    int grp_num = base_groups.size();
    for (int64_t i = 0; i < grp_num; i++) {
      if (!base_groups[i]->conv_cut_optimized &&
          base_groups[i]->_lgInfo.group_ops.size() > 1) {
        ilp_func_trace tmp_trace(
            llvm::formatv(
                "cut_this_group_is_better, i:{0}, is_cut_op_is_global:{1}", i,
                is_cut_op_is_global)
                .str(),
            base_groups[i]->_lgInfo.group_id);
        cut_this_group_is_better(*base_groups[i], pass_ir, base_groups,
                                 processed_ops, is_cut_op_is_global);
        all_optimized = false;
      }
      if (need_exit(processed_ops)) {
        return;
      }
    }
    if (all_optimized) {
      break;
    }
  }
}

static void l2m_process(ilp_LgInfo &sub_group,
                        std::vector<std::pair<Value, int64_t>> &value_size,
                        bool l2m_en) {
  LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::errs() << "process l2m...\n"; });
  auto l2mem_alloc_ptr = std::make_shared<l2mem_alloc>();
  sub_group.l2mem_alloc = l2mem_alloc_ptr;
  auto &map_l2m_load = sub_group.map_l2m_load;
  auto &grp_time_step = sub_group.timeStepPtrs;
  if (sub_group.p_special_grp) {
    if (l2m_en) {
      for (auto itr : grp_time_step[0]->vec_l2m_value_info) {
        auto name = module::getName(itr.value).str();
        uint32_t size = Arch::get_gmem_bytes(itr.value);
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::errs() << "alloc tensor:" << name << ", size:" << size
                       << " in l2m\n";
        });
        if (l2mem_alloc_ptr->alloc(0, name, itr.value, size)) {
          map_l2m_load[itr.load_ts].push_back(itr);
        }
      }
    }

    // If tp is deployed concurrently, reduce tasks need to be performed on l2m
    if (sub_group.shape_secs.h_slice_num > 1 &&
        module::getChip() == module::Chip::BM1690) {
      for (auto itr : sub_group.value_store_to_l2m) {
        auto name = module::getName(itr.first).str();
        uint32_t size = Arch::get_gmem_bytes(itr.first);
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::errs() << "alloc tensor:" << name << ", size:" << size
                       << " in l2m for store\n";
        });
        if (!l2mem_alloc_ptr->alloc(0, name, itr.first, size)) {
          assert(false);
        }
      }
    }
    return;
  }

  if (!l2m_en) {
    return;
  }
  int ts_count = grp_time_step[0]->ts_count;
  int core_num_per_pipe0 = grp_time_step[0]->ncdhw_steps.size();
  for (auto itr : grp_time_step[0]->vec_l2m_value_info) {
    // LAYER_GROUP_LOG_DEBUG_BLOCK({
    //   llvm::errs() << "check Value:" << module::getName(itr.value).str()
    //                << ", slice_idx:" << itr.slice_idx
    //                << ", pipe0 load ts:" << itr.load_ts << "\n";
    // });
    int parallel_core_num = core_num_per_pipe0;
    int min = itr.load_ts;
    // Traverse all other streams except the first stream, the first stream
    // being the longest
    for (int j = 1; j < grp_time_step.size(); j++) {
      parallel_core_num += grp_time_step[j]->ncdhw_steps.size();
      for (auto itr3 = grp_time_step[j]->vec_l2m_value_info.begin();
           itr3 != grp_time_step[j]->vec_l2m_value_info.end(); ++itr3) {
        if (itr3->value == itr.value && itr3->slice_idx == itr.slice_idx) {
          // LAYER_GROUP_LOG_DEBUG_BLOCK({
          //   llvm::errs() << "find in pipe:" << j
          //                << ", load ts:" << itr3->load_ts << "\n";
          // });
          if (itr3->load_ts < min) {
            min = itr3->load_ts;
          }
        }
      }
    }
    if (parallel_core_num > 1) {
      if (map_l2m_load.find(min) == map_l2m_load.end()) {
        map_l2m_load[min] = std::vector<l2m_value_info>();
      }
      map_l2m_load[min].push_back(itr);
    }
  }

  for (int m = -1; m < ts_count; m++) {
    if (map_l2m_load.find(m) != map_l2m_load.end()) {
      for (auto itr : map_l2m_load[m]) {
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::errs() << " Value:" << module::getName(itr.value).str()
                       << " slice_idx:" << itr.slice_idx << " load ts:" << m
                       << " free ts:" << itr.free_ts << "\n";
        });
      }
    }
  }

  int total_weight_size = 0, l2_mem_size = 128 * 1024 * 1024;
  int weight_num = value_size.size();
  for (auto it2 : value_size) {
    total_weight_size += it2.second;
  }
  std::vector<Value> value_l2m;
  if (total_weight_size > l2_mem_size) {
    int share_mem_size = 0;
    for (int i = weight_num - 1; i > 0; i--) {
      std::vector<std::pair<Value, int64_t>> value_size_l2m;
      std::vector<int64_t> value_l2m_addr;
      value_l2m.clear();
      share_mem_size += value_size[i].second;
      total_weight_size = 0;
      int addr = 0;
      for (auto it2 : value_size) {
        total_weight_size += it2.second;
        if (total_weight_size > l2_mem_size - (int)(share_mem_size * 1.5)) {
          break;
        }
        value_size_l2m.push_back(it2);
        value_l2m.push_back(it2.first);
        value_l2m_addr.push_back(addr);
        addr += it2.second;
      }
      l2mem_alloc_ptr->clear();
      for (auto it3 : value_size_l2m) {
        auto name = module::getName(it3.first).str();
        uint32_t size = Arch::get_gmem_bytes(it3.first);
        l2mem_alloc_ptr->alloc(-1, name, it3.first, size); // error, fix
                                                           // me!!!!!!
      }

      std::map<int, std::vector<l2m_value_info>> map_l2m_free;
      bool failed = false;
      for (int m = -1; m < ts_count; m++) {
        // Process the l2m tensor that needs to be released in this time slot
        if (map_l2m_free.find(m) != map_l2m_free.end()) {
          for (auto it3 : map_l2m_free[m]) {
            auto name = module::getName(it3.value).str();
            l2mem_alloc_ptr->free(it3.slice_idx, name);
          }
        }
        // Process the l2m tensor that needs to be allocated in this time slot
        if (map_l2m_load.find(m) != map_l2m_load.end()) {
          for (auto it3 : map_l2m_load[m]) {
            if (std::find(value_l2m.begin(), value_l2m.end(), it3.value) ==
                value_l2m.end()) {
              auto name = module::getName(it3.value).str();
              uint32_t size = Arch::get_gmem_bytes(it3.value);
              failed =
                  l2mem_alloc_ptr->alloc(it3.slice_idx, name, it3.value, size);
              if (failed) {
                break;
              }
              // Records the currently allocated l2m tensor time slot to be
              // released
              if (map_l2m_free.find(it3.free_ts) == map_l2m_free.end()) {
                map_l2m_free[it3.free_ts] = std::vector<l2m_value_info>();
              }
              map_l2m_free[it3.free_ts].push_back(it3);
            }
          }
        }
        if (failed) {
          break;
        }
      }
    }
  } else {
    LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::errs() << "l2m enough \n"; });
    for (auto it3 : value_size) {
      value_l2m.push_back(it3.first);
      auto name = module::getName(it3.first).str();
      uint32_t size = Arch::get_gmem_bytes(it3.first);
      l2mem_alloc_ptr->alloc(-1, name, it3.first, size);
    }
  }

  for (int m = -1; m < ts_count; m++) {
    if (map_l2m_load.find(m) != map_l2m_load.end()) {
      for (auto &itr : map_l2m_load[m]) {
        if (itr.slice_idx > 0 && std::find(value_l2m.begin(), value_l2m.end(),
                                           itr.value) != value_l2m.end()) {
          LAYER_GROUP_LOG_DEBUG_BLOCK({
            llvm::errs() << "value:" << module::getName(itr.value).str()
                         << ",set valid false\n";
          });
          itr.valid = false;
        }
      }
    }
  }
  // pass_ir->map_l2m_loads.push_back(map_l2m_load);
  // pass_ir->lg_l2mem_alloc_ptr.push_back(l2mem_alloc_ptr);
}

static void EliminatingDuplicatePipeline(ilp_LgInfo &ilp_sub_group) {
  std::vector<std::shared_ptr<ILPTimeStep>> timeStepPtrs_todo, tmp_timeStepPtrs,
      timeStepPtrs_unique;
  auto &timeStepPtrs = ilp_sub_group.timeStepPtrs;
  int size = timeStepPtrs.size();
  if (size > 0) {
    timeStepPtrs_todo.assign(timeStepPtrs.begin(), timeStepPtrs.end());
    do {
      tmp_timeStepPtrs.assign(timeStepPtrs_todo.begin(),
                              timeStepPtrs_todo.end());
      timeStepPtrs_todo.clear();
      for (int n = 1; n < tmp_timeStepPtrs.size(); n++) {
        if (tmp_timeStepPtrs[0]->IsSameWith(tmp_timeStepPtrs[n])) {
          for (auto itr : tmp_timeStepPtrs[n]->ncdhw_steps) {
            for (auto itr2 : itr.second) {
              tmp_timeStepPtrs[0]->addSliceNcdhwSteps(itr.first, itr2);
            }
          }
        } else {
          timeStepPtrs_todo.push_back(tmp_timeStepPtrs[n]);
        }
      }
      timeStepPtrs_unique.push_back(tmp_timeStepPtrs[0]);
      if (timeStepPtrs_todo.size() == 0) {
        break;
      }
    } while (true);
    // llvm::errs() <<"timeStepPtrs old size:"<<size<<", new
    // size:"<<timeStepPtrs_unique.size()<<"\n";
    timeStepPtrs.assign(timeStepPtrs_unique.begin(), timeStepPtrs_unique.end());
  }
}

static bool
is_same_pipeline(int core_id, ilp_LgInfo &ilp_sub_group,
                 TensorInfo &tensor_infos,
                 std::map<int64_t, std::vector<std::vector<int64_t>>> vec_ncdhw,
                 int core_slice_num) {
  auto sub_group = ilp_sub_group._lgInfo;
  auto &timeStepPtrs = ilp_sub_group.timeStepPtrs;
  bool all_slice_same = false;
  LAYER_GROUP_LOG_DEBUG_BLOCK(
      { llvm::errs() << "pipeline num:" << timeStepPtrs.size() << "\n"; });
  for (int n = 0; n < timeStepPtrs.size(); n++) {
    std::vector<std::vector<int64_t>> &ncdhw_steps =
        timeStepPtrs[n]->ncdhw_steps.begin()->second;
    if (ncdhw_steps.size() == core_slice_num) {
      all_slice_same = true;
      for (int m = 0; m < core_slice_num; m++) {
        std::vector<int64_t> &his_steps = ncdhw_steps[m];
        std::vector<int64_t> ncdhw = vec_ncdhw[core_id][m];
        if (tensor_infos.find(sub_group.group_ops[0]->getOperand(0)) ==
            tensor_infos.end()) {
          assert(false);
        }
        // todo Here it is used for single branches
        slice_info_t &slice_info =
            tensor_infos[sub_group.group_ops[0]->getOperand(0)].slice_info;
        // for (auto itr = tensor_infos.begin(); itr != tensor_infos.end();
        // ++itr) {
        if (slice_info.n[his_steps[0]].second !=
            slice_info.n[ncdhw[0]].second) {
          LAYER_GROUP_LOG_DEBUG_BLOCK({
            llvm::errs() << "slice n not equal:"
                         << slice_info.n[his_steps[0]].second << " vs "
                         << slice_info.n[ncdhw[0]].second << "\n";
          });
          all_slice_same = false;
          break;
        }
        if (slice_info.c[his_steps[1]].second !=
            slice_info.c[ncdhw[1]].second) {
          LAYER_GROUP_LOG_DEBUG_BLOCK({
            llvm::errs() << "slice c not equal:"
                         << slice_info.c[his_steps[1]].second << " vs "
                         << slice_info.c[ncdhw[1]].second << "\n";
          });
          all_slice_same = false;
          break;
        }
        if (slice_info.d[his_steps[2]].second !=
            slice_info.d[ncdhw[2]].second) {
          LAYER_GROUP_LOG_DEBUG_BLOCK({
            llvm::errs() << "slice d not equal:"
                         << slice_info.d[his_steps[2]].second << " vs "
                         << slice_info.d[ncdhw[2]].second << "\n";
          });
          all_slice_same = false;
          break;
        }
        if (slice_info.h[his_steps[3]].second !=
            slice_info.h[ncdhw[3]].second) {
          LAYER_GROUP_LOG_DEBUG_BLOCK({
            llvm::errs() << "slice h not equal:"
                         << slice_info.h[his_steps[3]].second << " vs "
                         << slice_info.h[ncdhw[3]].second << "\n";
          });
          all_slice_same = false;
          break;
        }
        if (slice_info.w[his_steps[4]].second !=
            slice_info.w[ncdhw[4]].second) {
          LAYER_GROUP_LOG_DEBUG_BLOCK({
            llvm::errs() << "slice w not equal:"
                         << slice_info.w[his_steps[4]].second << " vs "
                         << slice_info.w[ncdhw[4]].second << "\n";
          });
          all_slice_same = false;
          break;
        }
      }
      if (all_slice_same) {
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::errs() << "core " << core_id
                       << ",all slice shape same with pipeline " << n
                       << ", skip ILP\n";
        });
        for (int m = 0; m < core_slice_num; m++) {
          std::vector<int64_t> ncdhw = vec_ncdhw[core_id][m];
          timeStepPtrs[n]->addSliceNcdhwSteps(core_id, ncdhw);
        }
        break;
      }
    } else {
      LAYER_GROUP_LOG_DEBUG_BLOCK(
          { llvm::errs() << "pipeline slice num not equal\n"; });
    }
  }
  return all_slice_same;
}

static bool
backward_update_slice2(ilp_LgInfo &ilp_lg_info, const shape_secs_t &shape_secs,
                       const std::pair<Value, Operation *> &out,
                       std::list<std::pair<Value, Operation *>> &tensor_branchs,
                       TensorInfo &tensor_infos,
                       std::multiset<Operation *> &op_set,
                       const ValueSet &out_tensor_set) {
  int64_t n, c, d, h, w;
  auto lg_info = ilp_lg_info._lgInfo;
  // Don't backward when this out tensor is the input of the group
  if (std::find(lg_info.group_ins.begin(), lg_info.group_ins.end(),
                out.first) != lg_info.group_ins.end()) {
    // return check_hsecs(out, tensor_infos[out].slice_info, lg_info.type);
    return true;
  }
  auto op = out.first.getDefiningOp();
  if (isa<tpu::Conv2DOp>(op) && module::isBM1684Family()) {
    auto conv_attr = dyn_cast<tpu::Conv2DOp>(op).parseParam();
    if (conv_attr.use_3ic_optimize) {
      return false;
    }
  }
  auto mode = getRunMode(op);
  op_set.insert(op);

  slice_info_t &out_si = tensor_infos[out.first].slice_info;
  auto &group_ins = lg_info.group_ins;

  for (auto in : op->getOperands()) {
    slice_info_t si;
    auto pre_op = in.getDefiningOp();
    if (pre_op && isa<top::NoneOp>(pre_op)) {
      continue;
    }
    if (is_value_dont_split(in)) {
      module::getNCDHW(in, n, c, d, h, w, lg_info.type);
      si.n.emplace_back(std::pair(0, n));
      si.c.emplace_back(std::pair(0, c));
      si.d.emplace_back(std::pair(0, d));
      si.h.emplace_back(std::pair(0, h));
      si.w.emplace_back(std::pair(0, w));
      tensor_infos[in] = tensor_info_t(si);
      continue;
    }
    bool hold_in_lmem = false;
    bool is_group_in =
        std::find(group_ins.begin(), group_ins.end(), in) != group_ins.end();
    auto ret =
        get_backward_slice_info2(si, out_si, op, in, shape_secs, lg_info.type,
                                 hold_in_lmem, is_group_in);
    if (ret == false) {
      return false;
    }

    if (pre_op && isa<tpu::MaxPoolWithMaskOp>(pre_op)) {
      for (int j = 0; j < pre_op->getNumResults(); j++) {
        auto res = pre_op->getResult(j);
        if (res == in) {
          continue;
        }
        if (tensor_infos.find(res) != tensor_infos.end()) {
          tensor_infos[res] = tensor_info_t(op, si);
        }
      }
    }

    auto iter = tensor_infos.find(in);
    if (iter != tensor_infos.end()) {
      if (false == is_same_slice_info(si, iter->second.slice_info)) {
        if (module::isCV18xx() || mode == RunMode::TPU_DYNAMIC)
          return false;
        // only Conv2D allow differnece for now
        // if (pre_op) {
        //   for (auto user : pre_op->getUsers()) {
        //     if (isa<ReturnOp>(user)) {
        //       llvm::errs() << "skip ReturnOp\n";
        //       continue;
        //     }
        //     if (!(std::find(lg_info.group_ops.begin(),
        //     lg_info.group_ops.end(),
        //                     user) != lg_info.group_ops.end() &&
        //           isa<tpu::Conv2DOp>(user) &&
        //           module::isUniformQuantized(in))||
        //            lg_info.group_outs.size() != 1 ) {
        //       llvm::errs() << "xxx1\n";
        //       return false;
        //     }
        //   }
        // }
        bool is_hw_overlap = true;
        for (int i = 0; i < shape_secs.hsecs; i++) {
          is_hw_overlap *=
              std::max(si.h[i].first, iter->second.slice_info.h[i].first) <
              std::min(si.h[i].first + si.h[i].second,
                       iter->second.slice_info.h[i].first +
                           iter->second.slice_info.h[i].second);
        }
        for (int i = 0; i < shape_secs.wsecs; i++) {
          is_hw_overlap *=
              std::max(si.w[i].first, iter->second.slice_info.w[i].first) <
              std::min(si.w[i].first + si.w[i].second,
                       iter->second.slice_info.w[i].first +
                           iter->second.slice_info.w[i].second);
        }
        if (is_hw_overlap) {
          slice_info_t si_both;
          si_both.n = si.n;
          si_both.c = si.c;
          si_both.d = si.d;
          for (int i = 0; i < shape_secs.hsecs; i++) {
            int64_t h_lowest =
                std::min(si.h[i].first, iter->second.slice_info.h[i].first);
            int64_t h_highest =
                std::max(si.h[i].first + si.h[i].second,
                         iter->second.slice_info.h[i].first +
                             iter->second.slice_info.h[i].second);
            si_both.h.push_back(
                std::pair<int64_t, int64_t>(h_lowest, h_highest - h_lowest));
          }
          for (int i = 0; i < shape_secs.wsecs; i++) {
            int64_t w_lowest =
                std::min(si.w[i].first, iter->second.slice_info.w[i].first);
            int64_t w_highest =
                std::max(si.w[i].first + si.w[i].second,
                         iter->second.slice_info.w[i].first +
                             iter->second.slice_info.w[i].second);
            si_both.w.push_back(
                std::pair<int64_t, int64_t>(w_lowest, w_highest - w_lowest));
          }
          auto tmp = tensor_info_t(op, si_both);
          auto slice_infos = tensor_infos[in].slice_infos;
          for (auto itr = slice_infos.begin(); itr != slice_infos.end();
               ++itr) {
            tmp.add_slice_info(itr->first, itr->second);
          }
          tensor_infos[in] = tmp;
          tensor_infos[in].hold_in_lmem = hold_in_lmem;
          if (pre_op && isa<tpu::MaxPoolWithMaskOp>(pre_op)) {
            for (int j = 0; j < pre_op->getNumResults(); j++) {
              auto res = pre_op->getResult(j);
              if (res == in) {
                continue;
              }
              auto tmp = tensor_info_t(op, si_both);
              auto slice_infos = tensor_infos[res].slice_infos;
              for (auto itr = slice_infos.begin(); itr != slice_infos.end();
                   ++itr) {
                tmp.add_slice_info(itr->first, itr->second);
              }
              tensor_infos[res] = tmp;
            }
          }
        } else {
          return false;
        }
      } else {
        tensor_infos[in].add_slice_info(op, si);
      }
    } else {
      tensor_infos[in] = tensor_info_t(op, si);
      tensor_infos[in].hold_in_lmem = hold_in_lmem;
    }

    if (strip_back_judge2(in, lg_info, op_set, out_tensor_set)) {
      tensor_branchs.push_back(std::make_pair(in, op));
    }
  }
  return true;
}

int vec_index(const std::vector<Operation *> &group_ops, Operation *op) {
  int idx = 0;
  for (auto group_op : group_ops) {
    if (group_op == op) {
      return idx;
    }
    idx++;
  }
  assert(false);
  return 0;
}

static op_var_pos_info
findVarBound(const std::vector<op_var_pos_info> &op_var_bound,
             std::pair<int, int> key) {
  for (int i = 0, size = op_var_bound.size(); i < size; i++) {
    if (op_var_bound[i].key == key) {
      return op_var_bound[i];
    }
  }
  return op_var_pos_info();
}

static int
getTensorLmemBytes(Operation *op, Value &value, TensorInfo &tensor_infos,
                   const std::vector<int64_t> &ncdhw_idx,
                   ilp_LgInfo &ilp_lg_info, bool eu_align = true,
                   int64_t *slice_n = nullptr, int64_t *slice_c = nullptr,
                   int64_t *slice_d = nullptr, int64_t *slice_h = nullptr,
                   int64_t *slice_w = nullptr) {
  if (tensor_infos.find(value) == tensor_infos.end()) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({
      llvm::errs() << "value:" << module::getName(value).str()
                   << " has no slice_info\n";
    });
    assert(false);
  }

  slice_info_t &slice_info = tensor_infos[value].slice_info;
  if (module::isDebugCmdEnable("detail_info_show")) {
    llvm::errs() << "n size:" << slice_info.n.size()
                 << ", c size:" << slice_info.c.size()
                 << ", h size:" << slice_info.h.size() << "\n";
    for (auto [index, itr] : llvm::enumerate(slice_info.n)) {
      if (index == 0 || index == slice_info.n.size() - 1) {
        llvm::errs() << "n offset:" << itr.first << ", len:" << itr.second
                     << "\n";
      }
    }
    for (auto [index, itr] : llvm::enumerate(slice_info.c)) {
      if (index == 0 || index == slice_info.c.size() - 1) {
        llvm::errs() << "c offset:" << itr.first << ", len:" << itr.second
                     << "\n";
      }
    }
    for (auto [index, itr] : llvm::enumerate(slice_info.h)) {
      if (index == 0 || index == slice_info.h.size() - 1) {
        llvm::errs() << "h offset:" << itr.first << ", len:" << itr.second
                     << "\n";
      }
    }
    for (auto [index, itr] : llvm::enumerate(slice_info.w)) {
      if (index == 0 || index == slice_info.w.size() - 1) {
        llvm::errs() << "w offset:" << itr.first << ", len:" << itr.second
                     << "\n";
      }
    }
  }

  int c_idx = ncdhw_idx[1], h_idx = ncdhw_idx[3];
  if (ilp_lg_info.p_special_grp) {
    auto map_value_to_cut_dims =
        ilp_lg_info.p_special_grp->map_value_to_cut_dims;
    if (map_value_to_cut_dims.find(value) != map_value_to_cut_dims.end()) {
      auto dims = map_value_to_cut_dims[value];
      c_idx = ncdhw_idx[dims[1]];
      h_idx =
          ncdhw_idx[dims[3]]; // mlp的第2个matmul的右矩阵使用h索引作为c行索引
      llvm::errs() << "new c_idx:" << c_idx << ", h_idx:" << h_idx << "\n";
    }
  }
  int64_t n = slice_info.n[slice_info.n.size() == 1 ? 0 : ncdhw_idx[0]].second;
  int64_t c = slice_info.c[slice_info.c.size() == 1 ? 0 : c_idx].second;
  int64_t d = slice_info.d[slice_info.d.size() == 1 ? 0 : ncdhw_idx[2]].second;
  int64_t h = slice_info.h[slice_info.h.size() == 1 ? 0 : h_idx].second;
  int64_t w = slice_info.w[slice_info.w.size() == 1 ? 0 : ncdhw_idx[4]].second;
  if (slice_n)
    *slice_n = n;
  if (slice_c)
    *slice_c = c;
  if (slice_d)
    *slice_d = d;
  if (slice_h)
    *slice_h = h;
  if (slice_w)
    *slice_w = w;

  int in_lmem_bytes = align_64(backend::Arch::get_tensor_lmem_bytes(
      value, n, c, h, d, w, ilp_lg_info._lgInfo.type, eu_align));
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::errs() << "value_name:" << module::getName(value).str()
                 << ", shape:" << shape_str(n, c, d, h, w)
                 << ", size:" << in_lmem_bytes << "\n";
  });
  return in_lmem_bytes;
}

static int getOpLmemBytes(Operation *op, TensorInfo &tensor_infos,
                          const std::vector<int64_t> &ncdhw_idx,
                          ilp_LgInfo &ilp_lg_info, int &buffer_size) {
  int64_t in_n, in_c, in_d, in_h, in_w, out_n, out_c, out_d, out_h, out_w;
  auto ins = get_input_values(op);
  auto outs = get_output_values(op);
  auto op_name = replaceChars_for_dot(module::getName(op).str());
  int64_t in0_lmem_bytes =
      getTensorLmemBytes(op, ins[0], tensor_infos, ncdhw_idx, ilp_lg_info, true,
                         &in_n, &in_c, &in_d, &in_h, &in_w);
  int64_t in_lmem_bytes = in0_lmem_bytes;

  for (int i = 1; i < ins.size(); i++) {
    auto tmp_op = ins[i].getDefiningOp();
    if (tmp_op && isa<top::NoneOp>(tmp_op)) {
      continue;
    }
    in_lmem_bytes += getTensorLmemBytes(op, ins[i], tensor_infos, ncdhw_idx,
                                        ilp_lg_info, is_eu_align(ins[i]));
  }

  int64_t out0_lmem_bytes =
      getTensorLmemBytes(op, outs[0], tensor_infos, ncdhw_idx, ilp_lg_info,
                         true, &out_n, &out_c, &out_d, &out_h, &out_w);
  int64_t out_lmem_bytes = out0_lmem_bytes;
  for (int i = 1; i < outs.size(); i++) {
    out_lmem_bytes +=
        getTensorLmemBytes(op, outs[i], tensor_infos, ncdhw_idx, ilp_lg_info);
  }

  auto lg_op = cast<LocalGenInterface>(op);
  buffer_size = align_64(lg_op.getBufferSize(
      in0_lmem_bytes, out0_lmem_bytes, in_n, in_c, in_h, in_d, in_w, out_n,
      out_c, out_h, out_d, out_w, ilp_lg_info._lgInfo.type));
  int64_t used_mem = in_lmem_bytes + out_lmem_bytes + buffer_size;
  int64_t free_mem = backend::Arch::LMEM_BYTES - used_mem;
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::errs() << "buffer_size:" << buffer_size << ", free_mem:" << free_mem
                 << "\n";
  });
  return free_mem;
}

template <typename opTy>
static bool isOpTypeInGroup(const std::vector<Operation *> &group_ops,
                            std::vector<Operation *> &query_ops) {
  query_ops.clear();
  for (auto op : group_ops) {
    if (isa<opTy>(op)) {
      query_ops.push_back(op);
    }
  }
  return query_ops.size() ? true : false;
}

static void align_secs_to_core_num(ilp_LgInfo &sub_group,
                                   const shape_secs_t &max_shape_secs) {
  int64_t core_num = 1;
  if (dyn_cast<MultiCoreInterface>(BM168x::instance())) {
    core_num = module::getCoreNum();
  }
  if (core_num > 1) {
    int64_t secs = sub_group.shape_secs.get_sec_num();
    int new_secs = align(secs, core_num);
    int sz = new_secs - secs;
    if (sz > 0) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({
        llvm::errs() << "algin secs:" << secs << " to " << new_secs << "\n";
      });
      shape_secs_t suitable_shape_secs = sub_group.shape_secs;
      for (int m = 0; m < sz; m++) {
        if (sub_group.p_special_grp) {
          if (!sub_group.p_special_grp->update_shape_secs_for_ilp_group(
                  sub_group.shape_secs, max_shape_secs)) {
            break;
          }
        } else {
          if (!update_shape_secs_for_ilp_group(sub_group.shape_secs,
                                               max_shape_secs)) {
            break;
          }
        }
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::errs() << "update shape shape_secs:"
                       << sub_group.shape_secs.info() << '\n';
        });
        if (sub_group.shape_secs.get_sec_num() > new_secs) {
          break;
        }
        suitable_shape_secs = sub_group.shape_secs;
      }
      sub_group.shape_secs = suitable_shape_secs;
    }
  }
}

static bool failProcess_insertNonOp(ilp_LgInfo &ilp_lg_info, Operation *fail_op,
                                    bool &inc_secs, int nonOp_insert_mode = 0) {
  assert(nonOp_insert_mode >= 0 && nonOp_insert_mode <= 2);
  auto &failed_ops = ilp_lg_info.failed_ops;
  auto &ops = ilp_lg_info._lgInfo.group_ops;
  auto &backup_ops = ilp_lg_info.backup_ops;
  if (nonOp_insert_mode == 2) {
    ilp_lg_info.is_fail_op_in_grp = false;
    return false;
  }
  if (std::find(failed_ops.begin(), failed_ops.end(), fail_op) !=
      failed_ops.end()) {
    ops.assign(backup_ops.begin(), backup_ops.end());
    ilp_lg_info.is_fail_op_in_grp = false;
    return false;
  } else {
    backup_ops.assign(ops.begin(), ops.end());
    failed_ops.push_back(fail_op);
    auto it = std::find(ops.begin(), ops.end(), fail_op);
    if (it != ops.end()) {
      if (nonOp_insert_mode == 0) {
        ops.insert(++it, nullptr);
      } else if (nonOp_insert_mode == 1) {
        ops.insert(it, nullptr);
        ops.insert(++it, nullptr);
      }
    }
    inc_secs = false;
  }
  return true;
}

bool stripe_mine_idx_slice2(ilp_LgInfo &ilp_lg_info,
                            const shape_secs_t &shape_secs,
                            TensorInfo &tensor_infos, Operation *&fail_op) {
  auto lg_info = ilp_lg_info._lgInfo;
  if (lg_info.group_ops.size() == 1) {
    return true;
  }
  fail_op = nullptr;
  tensor_infos.clear();

  int64_t n, c, d, h, w;
  std::list<std::pair<Value, Operation *>> tensor_branchs;
  std::multiset<Operation *> op_set;
  std::set<Value, value_compare> out_tensor_set;

  for (auto out : lg_info.group_outs) {
    module::getNCDHW(out, n, c, d, h, w, lg_info.type);
    auto istype = module::getStorageType(lg_info.group_ins[0]);
    auto ostype = module::getStorageType(out);
    int64_t bitwidth = std::min(istype.getIntOrFloatBitWidth(),
                                ostype.getIntOrFloatBitWidth());
    slice_info_t si = get_out_slice_info(shape_secs, n, c, h, d, w, bitwidth);
    tensor_infos[out] = tensor_info_t(si);
    out_tensor_set.insert(out);
    tensor_branchs.push_back(std::make_pair(out, nullptr));
  }

  bool ret = false;
  while (!tensor_branchs.empty()) {
    auto out_tensor = tensor_branchs.front();
    tensor_branchs.pop_front();
    ret = backward_update_slice2(ilp_lg_info, shape_secs, out_tensor,
                                 tensor_branchs, tensor_infos, op_set,
                                 out_tensor_set);
    if (!ret) {
      fail_op = out_tensor.first.getDefiningOp();
      llvm::errs() << module::getName(fail_op).str()
                   << " backward_update_slice2 fail"
                   << "\n";
      return false;
    }
  }

  // for (auto itr: tensor_infos) {
  //   llvm::errs() <<"tensor:"<< module::getName(itr.first).str()
  //   <<",v:"<<itr.first.getImpl()<<"\n"; for (auto itr3:
  //   itr.second.slice_info.n) {
  //     llvm::errs() <<"n offset:"<< itr3.first<<", len:"<< itr3.second <<"\n";
  //   }
  //   for (auto itr3: itr.second.slice_info.c) {
  //     llvm::errs() <<"c offset:"<< itr3.first<<", len:"<< itr3.second <<"\n";
  //   }
  //   for (auto itr3: itr.second.slice_info.h) {
  //     llvm::errs() <<"h offset:"<< itr3.first<<", len:"<< itr3.second <<"\n";
  //   }
  // }

  return true;
}

bool backward_gen_ilp_var2(ilp_LgInfo &ilp_lg_info, TensorInfo &tensor_infos,
                           std::shared_ptr<CycleCalculator> cycle_calculator_,
                           ILPTimeStep &ilp_timeStep,
                           const std::vector<int64_t> &ncdhw_idx, int slice_idx,
                           std::vector<op_var_pos_info> &op_var_bound,
                           Operation *&failOp, int &failMode,
                           std::map<std::string, std::string> &node_labels,
                           int64_t &load_bytes_for_next_ts, bool l2m_en,
                           int max_ahead_or_delay_ts) {
  auto lg_info = ilp_lg_info._lgInfo;
  std::string tmpStr;
  auto ops = lg_info.group_ops;
  assert(ops.size() > 1);
  std::string slice_name =
      llvm::formatv("slice_{0}_{1}_{2}_{3}_{4}", ncdhw_idx[0], ncdhw_idx[1],
                    ncdhw_idx[2], ncdhw_idx[3], ncdhw_idx[4]);
  for (int cur_op_idx = ops.size() - 1; cur_op_idx >= 0; cur_op_idx--) {
    auto var_pos_info =
        findVarBound(op_var_bound, std::make_pair(slice_idx, cur_op_idx));
    auto op = ops[cur_op_idx];
    if (op == nullptr) {
      ilp_timeStep.addOpInfo(var_pos_info.ts_id, op, 0,
                             backend::Arch::LMEM_BYTES, 0);
      load_bytes_for_next_ts = 0;
      continue;
    }
    auto op_name = replaceChars_for_dot(module::getName(op).str());
    auto slice_pos_info =
        findVarBound(op_var_bound, std::make_pair(slice_idx, 0));
    LAYER_GROUP_LOG_DEBUG_BLOCK({
      llvm::errs() << "-------------------cur_op_idx: " << cur_op_idx
                   << ", op: " << show_op_info(op) << " ----\n";
    });
    int buffer_size = 0;
    failMode = 0;
    int64_t mem_size_for_load =
        getOpLmemBytes(op, tensor_infos, ncdhw_idx, ilp_lg_info, buffer_size);
    if (mem_size_for_load < 0) {
      LAYER_GROUP_LOG_DEBUG_BLOCK(
          { llvm::errs() << "error, mem_size_for_load < 0\n"; });
      failOp = op;
      failMode = 1;
      return false;
    }
    if (mem_size_for_load - load_bytes_for_next_ts < 0) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({
        llvm::errs() << "error, mem_size_for_load:" << mem_size_for_load
                     << ", load_bytes_for_next_ts:" << load_bytes_for_next_ts
                     << "\n";
      });
      failOp = op;
      failMode = 2;
      return false;
    }
    auto type = lg_info.type == GROUP_MM_OPT3 ? GROUP_MM : lg_info.type;
    int bdc_cycle =
        cycle_calculator_->getLocalLayerCycle(op, tensor_infos, type, true);
    tmpStr = "mem_size_for_load: " + std::to_string(mem_size_for_load) +
             ", buffer_size: " + std::to_string(buffer_size) +
             ", slice_idx:" + std::to_string(slice_idx);
    LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::errs() << tmpStr << "\n"; });
    node_labels[op_name] = tmpStr;
    ilp_timeStep.addOpInfo(var_pos_info.ts_id, op, buffer_size,
                           mem_size_for_load, bdc_cycle);

    load_bytes_for_next_ts = 0;
    for (OpOperand &opd : op->getOpOperands()) {
      int opd_idx = opd.getOperandNumber();
      auto in = op->getOperand(opd_idx);
      auto inOp = in.getDefiningOp();
      if (inOp && isa<top::NoneOp>(inOp)) {
        continue;
      }
      if (!inOp) {
        // Since the following ops uses nullptr as the vacant op, the nullptr of
        // the external parameter is changed to any value, as long as it cannot
        // be found in ops
        inOp = (Operation *)0x1111;
      }
      bool is_not_split = is_value_dont_split(in);
      int64_t lmem_bytes = getTensorLmemBytes(op, in, tensor_infos, ncdhw_idx,
                                              ilp_lg_info, is_eu_align(in));
      if (ilp_lg_info.p_special_grp &&
          ilp_lg_info.p_special_grp->name() == "attention_group" &&
          ilp_lg_info.shape_secs.h_slice_num > 1 && isa<tpu::MatMulOp>(op) &&
          ilp_lg_info.p_special_grp->ops.back() == op) {
        llvm::errs() << "inc res lmem_bytes for attention_grp\n";
        lmem_bytes *= 2;
      }
      std::string value_name = module::getName(in).str();
      auto itr = std::find(ops.begin(), ops.end(), inOp);
      if (itr == ops.end()) {
        tensor_info_t info;
        if (tensor_infos.find(in) != tensor_infos.end()) {
          info = tensor_infos[in];
        } else {
          assert(false);
        }
        info.mode2 |= TIMESTEP2_LOAD;
        ilp_timeStep.addTensorSize(in, slice_idx, lmem_bytes);
        load_bytes_for_next_ts += lmem_bytes;
        int dma_cycle = cycle_calculator_->getGdmaCycle(in, info, lg_info.type);
        if (ilp_lg_info.p_special_grp) {
          if (ilp_lg_info.value_load_to_l2m.find(in) !=
              ilp_lg_info.value_load_to_l2m.end()) {
            ilp_lg_info.value_load_to_l2m[in] = lmem_bytes;
            dma_cycle /= 4;
          }
        } else {
          if (l2m_en && is_not_split) {
            dma_cycle /= 4;
          }
        }

        ilp_timeStep.addTensorCycle(in, slice_idx, dma_cycle);
        std::vector<std::string> var_names;
        int ts_idx = var_pos_info.start_ts;
        for (; ts_idx < var_pos_info.ts_id; ts_idx++) {
          if (var_pos_info.ts_id - ts_idx > max_ahead_or_delay_ts) {
            continue;
          }
          std::string var_name;
          if (is_not_split) {
            var_name = llvm::formatv("x_weight_{0}_use_by_{1}_at_pos{2}_load_{"
                                     "3}bytes_{4}cycle_at_ts{5}_{6}",
                                     value_name.c_str(), op_name.c_str(),
                                     var_pos_info.ts_id, lmem_bytes, dma_cycle,
                                     ts_idx, slice_name.c_str())
                           .str();
          } else {
            var_name = llvm::formatv("x_grp_input_{0}_use_by_{1}_at_pos{2}_"
                                     "load_{3}bytes_{4}cycle_at_ts{5}_{6}",
                                     value_name.c_str(), op_name.c_str(),
                                     var_pos_info.ts_id, lmem_bytes, dma_cycle,
                                     ts_idx, slice_name.c_str())
                           .str();
          }
          // llvm::errs() << "define: "<<var_name<<"\n";
          auto op3 = (ts_idx < slice_pos_info.ts_id)
                         ? ops[0]
                         : ops[ts_idx - slice_pos_info.ts_id];
          if (op3) {
            node_labels[module::getName(op3).str()] =
                "LoadVarDefine: " + var_name;
          }
          ilp_timeStep.addBinaryVar(ts_idx, slice_idx, -1, var_name, in, info,
                                    lmem_bytes);
          var_names.push_back(var_name);
          ilp_timeStep.addTimestepGdmaCycle(ts_idx, dma_cycle, var_name);
          ilp_timeStep.addTimestepMemUse(ts_idx, lmem_bytes, var_names);
        }

        bool load_to_l2m = false;
        if (ilp_lg_info.p_special_grp) {
          if (slice_idx == 0 && ilp_lg_info.value_load_to_l2m.find(in) !=
                                    ilp_lg_info.value_load_to_l2m.end()) {
            load_to_l2m = true;
          }
        } else {
          if (is_not_split) {
            load_to_l2m = true;
          }
        }
        ilp_timeStep.addRowConstraint(var_pos_info.ts_id, in, var_names, false,
                                      load_to_l2m);
      } else {
        int producer_pos =
            slice_pos_info.ts_id + std::distance(ops.begin(), itr);
        if (producer_pos != var_pos_info.ts_id - 1) {
          // If equal, corresponding to the most general direct adjacent case,
          // there is no need to add, the front slot has been calculated // the
          // above 2 conditions are identical.
          load_bytes_for_next_ts += lmem_bytes;
        }
      }
    }

    std::vector<std::pair<int, MPVariable *>> coeff_var_items;
    for (int j = 0; j < op->getNumResults(); j++) {
      auto res = op->getResult(j);
      tensor_info_t &info = tensor_infos[res];
      std::string name = module::getName(res).str();
      std::string op_name = module::getName(op).str();
      LAYER_GROUP_LOG_DEBUG_BLOCK(
          { llvm::errs() << "process res name:" << name << "\n"; });
      int64_t lmem_bytes =
          getTensorLmemBytes(op, res, tensor_infos, ncdhw_idx, ilp_lg_info);
      assert(lmem_bytes > 0);
      if (ilp_lg_info.p_special_grp) {
        if (ilp_lg_info.p_special_grp->name() == "attention_group" &&
            ilp_lg_info.shape_secs.h_slice_num > 1 && isa<tpu::MatMulOp>(op) &&
            ilp_lg_info.p_special_grp->ops.back() == op) {
          llvm::errs() << "inc opd lmem_bytes for attention_grp\n";
          lmem_bytes *= 2;
        }
        if (ilp_lg_info.value_store_to_l2m.find(res) !=
            ilp_lg_info.value_store_to_l2m.end()) {
          ilp_lg_info.value_store_to_l2m[res] = lmem_bytes;
        }
      }
      llvm::errs() << "res lmem_bytes:" << lmem_bytes << "\n";
      ilp_timeStep.addTensorSize(res, slice_idx, lmem_bytes);
      std::map<int, Operation *> map_user_pos;
      bool have_grp_out = false;
      for (auto user : res.getUsers()) {
        auto itr = std::find(ops.begin(), ops.end(), user);
        if (itr != ops.end()) {
          int consumer_pos =
              slice_pos_info.ts_id + std::distance(ops.begin(), itr);
          map_user_pos[consumer_pos] = user;
        } else {
          have_grp_out = true;
        }
      }

      int full_slice_bytes =
          getTensorLmemBytes(op, res, tensor_infos, ncdhw_idx, ilp_lg_info);
      int pre_user_pos = var_pos_info.ts_id, dma_cycle;
      std::vector<std::string> all_store_varnames;
      std::string ada_var_name;
      int idx = 0;
      bool first_user = true;
      if (map_user_pos.size() >
          0) { // If there is no user in the group, do not enter the branch, and
               // then process the store.
        // std::map<MPVariable *, std::vector<std::pair<int, MPVariable *>>>
        // map_x_var_items;
        Operation *pre_user = nullptr;
        for (auto it :
             map_user_pos) { // In the most common case, there will be one user
          auto user_pos = it.first;
          auto user = it.second;
          llvm::errs() << "process " << idx << "th user, user_pos:" << user_pos
                       << " " << show_op_info(user)
                       << ", pre_user_pos:" << pre_user_pos << "\n";
          auto user_name = module::getName(user).str();
          MPVariable *reside_x = nullptr;
          if (user_pos - pre_user_pos >= 2) {
            ada_var_name = llvm::formatv(
                "ada_var_for_{0}_gen_by_{1}_at_pos{2}_use_by_{3}_at_pos{4}",
                name.c_str(), op_name, var_pos_info.ts_id, user_name, user_pos);
            // llvm::errs() << "define: "<<ada_var_name<<"\n";
            reside_x = ilp_timeStep.solver->MakeIntVar(0, 1, ada_var_name);
            // map_x_var_items[reside_x] = std::vector<std::pair<int, MPVariable
            // *>>();
            ilp_var_info var_info;
            var_info.ts_idx = pre_user_pos + 1;
            var_info.slice_idx = slice_idx;
            var_info.ilp_var = reside_x;
            ilp_timeStep.mapILPVarInfo[ada_var_name] = var_info;

            info.mode2 = TIMESTEP2_ONLY_RESIDE;
            ts_var_t tmp;
            tmp.varName = ada_var_name;
            tmp.value = res;
            tmp.info = info;
            tmp.lmem_bytes = full_slice_bytes;
            tmp.slice_idx = slice_idx;
            std::vector<std::string> varNames;
            varNames.push_back(ada_var_name);
            // All tensor memory usage that is not controlled by op itself
            // and variables should be explicitly set mem_contrains; ada_var
            // does not need to set cycle_contrains
            for (int ts_idx = pre_user_pos + 1; ts_idx < user_pos; ts_idx++) {
              ilp_timeStep.timestep_table_[ts_idx].vec_ts_var.push_back(tmp);
              ilp_timeStep.addTimestepMemUse(ts_idx, full_slice_bytes,
                                             varNames);
              auto tmp = ops[ts_idx - slice_pos_info.ts_id];
              if (tmp) {
                node_labels[module::getName(tmp).str()] =
                    "add " + ada_var_name + " to mem_contrains";
              }
            }
          }

          if (user_pos - pre_user_pos >= 4) {
            info.mode2 = TIMESTEP2_STORE_AND_LOAD;
            dma_cycle = cycle_calculator_->getGdmaCycle(
                res, info, lg_info.type); // nullptr, 0
            ilp_timeStep.addTensorCycle(res, slice_idx, dma_cycle);
            llvm::errs() << "full_slice_bytes:" << full_slice_bytes
                         << ", dma_cycle:" << dma_cycle << "\n";
            std::vector<std::string> var_names;
            std::vector<std::pair<std::string, int>> store_var_names;
            llvm::errs() << "define store_var, ts_idx from " << pre_user_pos + 1
                         << " to " << user_pos << "\n";
            for (int ts_idx = pre_user_pos + 1; ts_idx < user_pos; ts_idx++) {
              std::string var_name = llvm::formatv(
                  "x_tensor_{0}_gen_at_pos{1}_store_{2}byte_{3}cycle_at_ts{4}_"
                  "use_by_{5}_at_pos{6}_{7}",
                  name.c_str(), var_pos_info.ts_id, full_slice_bytes, dma_cycle,
                  ts_idx, user_name, user_pos, slice_name.c_str());
              var_names.push_back(var_name);
              if (var_names.size() >= max_ahead_or_delay_ts) {
                break;
              }
            }
            for (int ts_idx = pre_user_pos + 1, offset = 0; ts_idx < user_pos;
                 ts_idx++) {
              std::string var_name = llvm::formatv(
                  "x_tensor_{0}_gen_at_pos{1}_store_{2}byte_{3}cycle_at_ts{4}_"
                  "use_by_{5}_at_pos{6}_{7}",
                  name.c_str(), var_pos_info.ts_id, full_slice_bytes, dma_cycle,
                  ts_idx, user_name, user_pos, slice_name.c_str());
              llvm::errs() << "  AdaLoadDefine: " << var_name << "\n";
              auto tmp = ops[ts_idx - slice_pos_info.ts_id];
              if (tmp) {
                node_labels[module::getName(tmp).str()] =
                    "AdaStoreDefine: " + var_name;
              }
              ilp_timeStep.addBinaryVar(ts_idx, slice_idx, 0, var_name, res,
                                        info, full_slice_bytes);
              ilp_timeStep.addTimestepGdmaCycle(ts_idx, dma_cycle, var_name);
              std::vector<std::string> var_names2;
              for (int n = offset++; n < var_names.size(); n++) {
                var_names2.push_back(var_names[n]);
              }
              store_var_names.push_back(std::make_pair(var_name, ts_idx));
              all_store_varnames.push_back(var_name);
              ilp_timeStep.addTimestepMemUse(ts_idx, lmem_bytes, var_names2);
              // if (!first_user) {
              //   for (auto itr = map_x_var_items.begin(); itr !=
              //   map_x_var_items.end(); ++itr) {
              //     itr->second.push_back(std::make_pair(1,
              //     ilp_timeStep.getMPVarByName(var_name)));
              //   }
              // }
              if (offset >= max_ahead_or_delay_ts) {
                break;
              }
            }

            dma_cycle = cycle_calculator_->getGdmaCycle(res, info, lg_info.type,
                                                        user, 1);
            var_names.clear();
            std::vector<std::pair<std::string, int>> load_var_names;
            llvm::errs() << "define load_var, ts_idx from " << pre_user_pos + 1
                         << " to " << user_pos << "\n";
            for (int ts_idx = pre_user_pos + 1; ts_idx < user_pos; ts_idx++) {
              if (user_pos - ts_idx > max_ahead_or_delay_ts) {
                continue;
              }
              std::string var_name = llvm::formatv(
                  "x_tensor_{0}_gen_by_{1}_at_pos{2}_load_{3}bytes_{4}cycle_at_"
                  "ts{5}_use_by_{6}_at_pos{7}_{8}",
                  name.c_str(), op_name, var_pos_info.ts_id, full_slice_bytes,
                  dma_cycle, ts_idx, user_name.c_str(), user_pos,
                  slice_name.c_str());
              // llvm::errs() << "  define: "<<var_name<<"\n";
              auto tmp = ops[ts_idx - slice_pos_info.ts_id];
              if (tmp) {
                node_labels[module::getName(tmp).str()] =
                    "AdaLoadDefine: " + var_name;
              }
              ilp_timeStep.addBinaryVar(ts_idx, slice_idx, 1, var_name, res,
                                        info, full_slice_bytes);
              var_names.push_back(var_name);
              load_var_names.push_back(std::make_pair(var_name, ts_idx));
              ilp_timeStep.addTimestepGdmaCycle(ts_idx, dma_cycle, var_name);
              ilp_timeStep.addTimestepMemUse(ts_idx, full_slice_bytes,
                                             var_names);
            }

            coeff_var_items.clear();
            for (auto store_var : store_var_names) {
              coeff_var_items.push_back(std::make_pair(
                  1, ilp_timeStep.getMPVarByName(store_var.first)));
            }
            for (auto load_var : load_var_names) {
              coeff_var_items.push_back(std::make_pair(
                  -1, ilp_timeStep.getMPVarByName(load_var.first)));
            }
            ilp_timeStep.addConstraint(
                0, 0, coeff_var_items); //定义sum(store_var) == sum(load_var)

            // for (auto itr = map_x_var_items.begin(); itr !=
            // map_x_var_items.end(); ++itr) {
            //   itr->second.push_back(std::make_pair(-1, itr->first));
            //   ilp_timeStep.addConstraint(0, 0, itr->second);
            // }

            coeff_var_items.clear();
            for (auto load_var : load_var_names) {
              coeff_var_items.push_back(std::make_pair(
                  1, ilp_timeStep.getMPVarByName(load_var.first)));
            }
            coeff_var_items.push_back(std::make_pair(1, reside_x));
            ilp_timeStep.addConstraint(
                1, 1, coeff_var_items); // define sum(load_var) + ada_var = 1

            coeff_var_items.clear();
            for (auto load_var : load_var_names) {
              coeff_var_items.push_back(
                  std::make_pair(load_var.second,
                                 ilp_timeStep.getMPVarByName(load_var.first)));
            }
            for (auto store_var : store_var_names) {
              coeff_var_items.push_back(
                  std::make_pair(-1 * store_var.second,
                                 ilp_timeStep.getMPVarByName(store_var.first)));
            }
            coeff_var_items.push_back(std::make_pair(2, reside_x));
            // define 2*ada_var + sum(pos*load_var) - sum(pos*store_var) >= 2
            ilp_timeStep.addConstraint(2, MPSolver::infinity(),
                                       coeff_var_items);
          } else {
            if (reside_x) {
              coeff_var_items.clear();
              coeff_var_items.push_back(std::make_pair(1, reside_x));
              ilp_timeStep.addConstraint(1, 1, coeff_var_items);
            }
            if (pre_user) {
              assert(idx != 0);
              ilp_timeStep.resideOpInValue(pre_user, res);
            }
          }
          pre_user_pos = user_pos;
          pre_user = user;
          idx++;
          first_user = false;
        }

        if (all_store_varnames.size() > 0) {
          if (have_grp_out) {
            coeff_var_items.clear();
            for (auto store_var : all_store_varnames) {
              coeff_var_items.push_back(
                  std::make_pair(1, ilp_timeStep.getMPVarByName(store_var)));
            }
            ilp_timeStep.addConstraint(1, 1, coeff_var_items);
            have_grp_out = false;
          } else {
            ilp_timeStep.addNewOutIntoReturnOp(all_store_varnames, res);
          }
        }
      }

      if (have_grp_out) {
        info.mode2 = TIMESTEP2_STORE;
        int64_t lmem_bytes =
            getTensorLmemBytes(op, res, tensor_infos, ncdhw_idx, ilp_lg_info);
        int dma_cycle =
            cycle_calculator_->getGdmaCycle(res, info, lg_info.type);
        std::vector<std::string> var_names;
        int ts_idx = pre_user_pos + 1;
        int end_idx = var_pos_info.end_ts + 1;
        if (ts_idx >= end_idx) {
          end_idx = ts_idx + 1;
        }
        int ts_idx2 = ts_idx;
        for (; ts_idx < end_idx; ts_idx++) {
          std::string var_name = llvm::formatv(
              "x_tensor_{0}_gen_at_pos{1}_store_{2}byte_{3}cycle_at_ts{4}_{5}",
              name.c_str(), var_pos_info.ts_id, lmem_bytes, dma_cycle, ts_idx,
              slice_name.c_str());
          auto op3 = (ts_idx - slice_pos_info.ts_id > ops.size() - 1)
                         ? ops[ops.size() - 1]
                         : ops[ts_idx - slice_pos_info.ts_id];
          if (op3) {
            node_labels[module::getName(op3).str()] =
                "StoreDefine: " + var_name;
          }
          var_names.push_back(var_name);
          if (var_names.size() >= max_ahead_or_delay_ts) {
            break;
          }
        }
        ts_idx = ts_idx2;
        for (int offset = 0; ts_idx < end_idx; ts_idx++) {
          std::string var_name = llvm::formatv(
              "x_tensor_{0}_gen_at_pos{1}_store_{2}byte_{3}cycle_at_ts{4}_{5}",
              name.c_str(), var_pos_info.ts_id, lmem_bytes, dma_cycle, ts_idx,
              slice_name.c_str());
          ilp_timeStep.addBinaryVar(ts_idx, slice_idx, -1, var_name, res, info,
                                    lmem_bytes);
          ilp_timeStep.addTimestepGdmaCycle(ts_idx, dma_cycle, var_name);
          std::vector<std::string> var_names2;
          for (int n = offset++; n < var_names.size(); n++) {
            var_names2.push_back(var_names[n]);
          }
          ilp_timeStep.addTimestepMemUse(ts_idx, lmem_bytes, var_names2);
          if (offset >= max_ahead_or_delay_ts) {
            break;
          }
        }
        if (var_names.size() > 0) {
          ilp_timeStep.addRowConstraint(pre_user_pos, res, var_names, true,
                                        false);
        }
      }
    }
  }
  return true;
}

static bool
ilp_for_single_group(LgPassIR *pass_ir, ilp_LgInfo &sub_group,
                     int &fail_process_mode, Operation *&fail_op,
                     std::shared_ptr<CycleCalculator> cycle_calculator_) {
  auto &ops = sub_group._lgInfo.group_ops;
  auto tmp_dot_graph_log = createSubnetGraph(ops);
  for (auto [index, op] : llvm::enumerate(ops)) {
    if (op) {
      tmp_dot_graph_log->add_node_label(module::getName(op).str(),
                                        "grp_ts" + std::to_string(index) + "*");
    }
  }

  bool ret = false;
  std::string tmpStr;
  ilp_func_trace tmp_trace(__func__, sub_group._lgInfo.group_id,
                           tmp_dot_graph_log);
  fail_op = nullptr;
  show_group(&sub_group._lgInfo);
  std::vector<std::pair<Operation *, int>> vec_op_hwsecs;
  shape_secs_t max_shape_secs;
  if (!sub_group.p_special_grp) {
    max_shape_secs = get_group_max_secs(sub_group._lgInfo, vec_op_hwsecs);
    tmpStr = shape_str(max_shape_secs.nsecs, max_shape_secs.csecs,
                       max_shape_secs.dsecs, max_shape_secs.hsecs,
                       max_shape_secs.wsecs);
    LAYER_GROUP_LOG_DEBUG_BLOCK(
        { llvm::errs() << "max_shape_secs:" << tmpStr << '\n'; });
    tmp_dot_graph_log->add_node_label("global_info",
                                      "max_shape_secs:" + tmpStr);
  }
  auto &shape_secs = sub_group.shape_secs;
  std::vector<std::pair<Value, int64_t>> value_size;
  int64_t core_num = dyn_cast<MultiCoreInterface>(BM168x::instance())
                         ? module::getCoreNum()
                         : 1;
  if (sub_group.p_special_grp) {
    if (!sub_group.p_special_grp->CalcMatMulGroupTpNum(sub_group, fail_op,
                                                       core_num)) {
      LAYER_GROUP_LOG_DEBUG_BLOCK(
          { llvm::errs() << "CalcMatMulGroupTpNum fail\n"; });
      // tmp_trace.need_exit();
      sub_group.is_fail_op_in_grp = false;
      return false;
    }
  } else {
    std::sort(vec_op_hwsecs.begin(), vec_op_hwsecs.end(),
              pair_op_int_Sort_by_int);
    std::vector<std::pair<Operation *, int>> vec_op_cut_secs;
    get_op_cut_sec_num(sub_group, vec_op_cut_secs);
    std::sort(vec_op_cut_secs.begin(), vec_op_cut_secs.end(),
              pair_op_int_Sort_by_int);
    // llvm::errs() << "vec_op_cut_secs:\n";
    // for (auto itr: vec_op_cut_secs) {
    //   llvm::errs() <<show_op_info(itr.first)<< ", count:"<<itr.second<<"\n";
    // }
    if (!init_group_data_secs2(sub_group, shape_secs, value_size, fail_op,
                               tmp_dot_graph_log, sub_group.options_)) {
      sub_group.is_fail_op_in_grp = false;
      if (!fail_op) {
        fail_op = vec_op_cut_secs.back().first;
      }
      tmpStr = module::getName(fail_op).str();
      LAYER_GROUP_LOG_DEBUG_BLOCK({
        llvm::errs() << "init_group_data_secs2 fail, will del op:" << tmpStr
                     << "\n";
      });
      tmp_dot_graph_log->add_node_label(tmpStr, "init_group_data_secs2 fail");
      // tmp_trace.need_exit();
      return false;
    }
    align_secs_to_core_num(sub_group, max_shape_secs);
  }

  tmpStr = shape_secs.info();
  LAYER_GROUP_LOG_DEBUG_BLOCK(
      { llvm::errs() << "init shape_secs:" << tmpStr << '\n'; });
  tmp_dot_graph_log->add_node_label("global_info", "init shape_secs:" + tmpStr);
  std::sort(value_size.begin(), value_size.end(), Sort_by_int);
  int slice_try_count = 0, nonOp_insert_mode,
      max_slice_cut_count = ops.size() > 10 ? 1 : 3;
  auto &tensor_infos = sub_group.tensor_infos;
  bool l2m_switch = module::isDebugCmdEnable("disable_l2m") ? false : true,
       inc_secs = true;
  if (module::getChip() != module::Chip::BM1690) {
    l2m_switch = false;
  }
  tmp_dot_graph_log->add_node_label("global_info", "enable_l2m");

  while (true) {
    if (inc_secs && ++slice_try_count > max_slice_cut_count) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::errs() << "layer group fail\n"; });
      return false; // Set to global layer
    }
    if (!inc_secs) {
      inc_secs = true;
    }
    int64_t secs = shape_secs.get_sec_num();
    bool l2m_en = l2m_switch && secs > 1 && core_num > 1;
    LAYER_GROUP_LOG_DEBUG_BLOCK({
      llvm::errs() << "shape_secs:" << shape_secs.info()
                   << " slice_try_count:" << slice_try_count
                   << " l2m_en:" << l2m_en << "\n";
    });
    int max_group_cycle = 0;
    sub_group.timeStepPtrs.clear();
    do {
      if (!sub_group.p_special_grp) {
        // tmp_dot_graph_log->export_dot("stripe_mine_idx_slice2_before");
        ret = stripe_mine_idx_slice2(sub_group, shape_secs, tensor_infos,
                                     fail_op);
        if (!ret) {
          tmpStr = module::getName(fail_op).str();
          tmp_dot_graph_log->add_node_label(tmpStr,
                                            "stripe_mine_idx_slice2 fail");
          LAYER_GROUP_LOG_DEBUG_BLOCK({
            llvm::errs() << "stripe_mine_idx_slice2 fail at" << tmpStr << "\n";
          });
          if (isa<tpu::UpsampleOp>(fail_op) && shape_secs.hsecs > 1) {
            // sub_group.is_fail_op_in_grp = false;
            fail_process_mode = 2;
            return false;
          }
          if (sub_group._cur_strategy != STRATEGY_SLICE_CUT_FIRST) {
            return false;
          } else {
            break;
          }
        }
        update_tensor_infos(sub_group._lgInfo, tensor_infos, shape_secs,
                            sub_group.p_special_grp ? 1 : 0);
        sub_group._lgInfo.update_bank_info();
      }

      std::map<int64_t, std::vector<std::vector<int64_t>>> vec_ncdhw;
      if (tensor_infos.find(sub_group._lgInfo.group_ops[0]->getOperand(0)) ==
          tensor_infos.end()) {
        assert(false);
      }
      auto &slice_info =
          tensor_infos[sub_group._lgInfo.group_ops[0]->getOperand(0)]
              .slice_info;
      get_sec_per_cores(sub_group, vec_ncdhw, core_num, slice_info);
      for (int core_id = 0; core_id < core_num; core_id++) {
        LAYER_GROUP_LOG_DEBUG_BLOCK(
            { llvm::errs() << "cur_core_id:" << core_id << "\n"; });
        int core_slice_num = vec_ncdhw[core_id].size();
        if (core_slice_num == 0) {
          break;
        }

        if (is_same_pipeline(core_id, sub_group, tensor_infos, vec_ncdhw,
                             core_slice_num)) {
          continue;
        }

        int slice_idx = 0, failMode = 0;
        std::vector<op_var_pos_info> op_var_bound;
        if (sub_group.p_special_grp) {
          op_var_bound =
              createOverlapStrategy(sub_group._lgInfo, core_slice_num, 1, 2, 2);
        } else {
          op_var_bound =
              createOverlapStrategy(sub_group._lgInfo, core_slice_num);
        }
        std::map<std::string, std::string> node_labels;
        auto ilp_timeStep = std::make_shared<ILPTimeStep>(
            sub_group._lgInfo, tmp_dot_graph_log, core_slice_num);
        int64_t load_bytes_for_next_ts = 0;
        while (slice_idx < core_slice_num) {
          std::vector<int64_t> ncdhw = vec_ncdhw[core_id][slice_idx];
          ilp_timeStep->addSliceNcdhwSteps(core_id, ncdhw);
          LAYER_GROUP_LOG_DEBUG_BLOCK({
            llvm::errs() << "slice" << slice_idx
                         << ", ncdhw:" << shape_str(ncdhw) << "\n";
          });
          ret = backward_gen_ilp_var2(
              sub_group, tensor_infos, cycle_calculator_, *ilp_timeStep, ncdhw,
              slice_idx, op_var_bound, fail_op, failMode, node_labels,
              load_bytes_for_next_ts, l2m_en, 4);
          if (!ret) {
            if (failMode == 1) {
              sub_group.is_fail_op_in_grp = false;
              return false;
            } else {
              if (!failProcess_insertNonOp(sub_group, fail_op, inc_secs)) {
                return false;
              }
            }
            break;
          }
          slice_idx++;
        }
        if (core_id == 0) {
          for (auto itr2 : node_labels) {
            tmp_dot_graph_log->add_node_label(itr2.first, itr2.second);
          }
        }
        if (!ret) {
          break;
        }

        bool merged = false;
        ilp_timeStep->merge_small_cycle_op(tensor_infos, merged,
                                           tmp_dot_graph_log);
        ilp_timeStep->prepare(tensor_infos);
        // tmp_dot_graph_log->export_dot("merge_small_cycle_op_after", true);
        ret = ilp_timeStep->run(fail_op);
        if (!ret) {
          auto error_info =
              "ilp_timeStep run fail, for core_id:" + std::to_string(core_id);
          LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::errs() << error_info << "\n"; });
          tmp_dot_graph_log->add_node_label(
              fail_op ? module::getName(fail_op).str() : "global_info",
              error_info);
          if (!failProcess_insertNonOp(sub_group, fail_op, inc_secs)) {
            return false;
          }
          // fail_process_mode = 1;
          break;
        }

        mem_alloc_status alloc_status;
        ret = ilp_timeStep->mem_alloc(alloc_status, value_size, tensor_infos,
                                      fail_op, nonOp_insert_mode);
        if (!ret) {
          auto error_info = "ilp_timeStep mem_alloc fail, for core_id:" +
                            std::to_string(core_id);
          LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::errs() << error_info << "\n"; });
          tmp_dot_graph_log->add_node_label(
              fail_op ? module::getName(fail_op).str() : "global_info",
              error_info);
          if (!failProcess_insertNonOp(sub_group, fail_op, inc_secs,
                                       nonOp_insert_mode)) {
            return false;
          }
          break;
        }

        int group_cycle, group_cycle_diff;
        std::vector<ts_cycle_info> ts_cycle;
        ilp_timeStep->get_group_cycle_info(group_cycle, group_cycle_diff,
                                           ts_cycle);
        if (group_cycle > max_group_cycle) {
          max_group_cycle = group_cycle;
        }
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::errs() << "core" << core_id << " group_cycle:" << group_cycle
                       << ", mem_alloc success\n";
        });
        tmp_dot_graph_log->add_node_label(
            "global_info", "core" + std::to_string(core_id) +
                               ", group_cycle:" + std::to_string(group_cycle) +
                               ", mem_alloc success");
        sub_group.timeStepPtrs.push_back(ilp_timeStep);
      }
      EliminatingDuplicatePipeline(sub_group);
    } while (false);

    if (ret) {
      LAYER_GROUP_LOG_DEBUG_BLOCK(
          { llvm::errs() << "ilp_timeStep success\n"; });
      tmp_dot_graph_log->add_node_label("global_info", "ilp_timeStep success");
      if (fail_process_mode == 1) {
        return true;
      }
      sub_group.group_cycle = max_group_cycle;
      l2m_process(sub_group, value_size, l2m_en);
      break;
    } else {
      if (!inc_secs) {
        continue;
      }
      if (sub_group.p_special_grp) {
        if (!sub_group.p_special_grp->update_shape_secs_for_ilp_group(
                sub_group.shape_secs, max_shape_secs)) {
          return false;
        }
      } else {
        if (!update_shape_secs_for_ilp_group(sub_group.shape_secs,
                                             max_shape_secs)) {
          return false;
        }
      }
      align_secs_to_core_num(sub_group, max_shape_secs);
      ops.erase(std::remove_if(ops.begin(), ops.end(),
                               [](Operation *op) { return op == nullptr; }),
                ops.end());
      LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::errs() << "update_shape_secs\n"; });
      tmp_dot_graph_log->add_node_label("global_info", "update_shape_secs");
    }
  }

  return true;
}

void GroupMethod::init_ilp_base_groups(LgPassIR *pass_ir) {
  get_base_dfs_topo_groups(pass_ir->tmp_base_groups);
}
int ilp_LgInfo::group_count = 1;
void ilp_LgInfo::save_result(LgPassIR *pass_ir) {
  if (!grp_is_valid(_lgInfo.group_ops)) {
    return;
  }

  pass_ir->ILP_time_steps.push_back(timeStepPtrs);
  pass_ir->shape_secs.push_back(shape_secs);

  pass_ir->lg_tensor_infos_.push_back(tensor_infos);
  pass_ir->lg_infos.push_back(_lgInfo);
  pass_ir->group_cycles.push_back(group_cycle);
  pass_ir->map_l2m_loads.push_back(map_l2m_load);
  pass_ir->lg_l2mem_alloc_ptr.push_back(l2mem_alloc);
}

std::shared_ptr<ilp_LgInfo>
ilp_LgInfo::high_solver(LgPassIR *pass_ir,
                        std::shared_ptr<CycleCalculator> cycle_calculator_) {
  auto ops_ori = _lgInfo.group_ops;
  _cur_strategy = STRATEGY_GROUP_CUT_FIRST;
  LAYER_GROUP_LOG_DEBUG_BLOCK(
      { llvm::errs() << "ilp_debug: STRATEGY_GROUP_CUT_FIRST test\n"; });
  base_solver(pass_ir, cycle_calculator_);

  // if (module::isDebugCmdEnable("opt3_o2")) { }

  if (module::isDebugCmdEnable("opt3_o3")) {
    LAYER_GROUP_LOG_DEBUG_BLOCK(
        { llvm::errs() << "ilp_debug: STRATEGY_SLICE_CUT_FIRST test\n"; });
    auto ilp_cloned =
        CreateIlpLgInfo(ops_ori, options_, STRATEGY_SLICE_CUT_FIRST);
    ilp_cloned->base_solver(pass_ir, cycle_calculator_);
    if (group_cycle > ilp_cloned->group_cycle) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({
        llvm::errs() << "ilp_debug:strategy STRATEGY_SLICE_CUT_FIRST better, "
                     << group_cycle << " vs " << ilp_cloned->group_cycle
                     << "\n";
      });
      return ilp_cloned;
    } else {
      LAYER_GROUP_LOG_DEBUG_BLOCK({
        llvm::errs() << "ilp_debug:strategy STRATEGY_GROUP_CUT_FIRST better, "
                     << group_cycle << " vs " << ilp_cloned->group_cycle
                     << "\n";
      });
    }
  }
  return nullptr;
}

void ilp_LgInfo::base_solver(
    LgPassIR *pass_ir, std::shared_ptr<CycleCalculator> cycle_calculator_) {
  auto &ops = _lgInfo.group_ops;
  ilp_func_trace tmp_trace(__func__);
  int fail_process_mode = 0;
  Operation *fail_op = nullptr;
  _lgInfo.update_group_io(options_.opt);
  std::map<Operation *, bool> break_op_reside;
  std::map<Operation *, bool> *break_op_reside_ptr = &break_op_reside;
  std::vector<Operation *> break_ops, excluded_ops;
  auto ret = ilp_for_single_group(pass_ir, *this, fail_process_mode, fail_op,
                                  cycle_calculator_);
  if (!ret) {
    if (_cur_strategy == STRATEGY_SEARCH_CONV_CUT) {
      return; //搜索模式下不再嵌套group
    }
    if (fail_op && fail_process_mode == 0) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({
        llvm::errs() << "ilp_debug: ilp_for_single_group fail_op:"
                     << show_op_info(fail_op) << "\n";
      });
      break_op_reside[fail_op] = is_fail_op_in_grp;
      break_ops.push_back(fail_op);
      if (!is_fail_op_in_grp) {
        global_layers.push_back(fail_op);
      }
    } else if (fail_process_mode == 2) {
      LAYER_GROUP_LOG_DEBUG_BLOCK(
          { llvm::errs() << "ilp_debug: fail_process_mode 2\n"; });
      if (isOpTypeInGroup<tpu::UpsampleOp>(ops, break_ops)) {
        for (auto op : break_ops) {
          LAYER_GROUP_LOG_DEBUG_BLOCK({
            llvm::errs() << "ilp_debug: break_op:" << show_op_info(op) << "\n";
          });
          global_layers.push_back(op);
        }
      }
    } else {
      global_layers.assign(ops.begin(), ops.end());
    }
  } else {
    group_success = true;
    return;
  }

  if (break_ops.size() > 0) {
    for (auto [i, grp] : llvm::enumerate(seg_grp_ops_by_global_op(
             ops, break_ops, excluded_ops, options_, break_op_reside_ptr))) {
      if (grp.size() > 1) {
        ilp_func_trace tmp_trace(
            llvm::formatv("ilp_debug: process_sub_group, i:{0}", i).str());
        auto sub_ops = sortOpsByOtherOpsOrder(_lgInfo.group_ops, grp);
        auto tmpLgInfo = CreateIlpLgInfo(sub_ops, options_);
        if (p_special_grp) {
          tmpLgInfo->p_special_grp = p_special_grp;
          if (!p_special_grp->convert_to_other_type(sub_ops,
                                                    tmpLgInfo->p_special_grp)) {
            LAYER_GROUP_LOG_DEBUG_BLOCK(
                {
                  llvm::errs()
                      << "ilp_debug: matmul grp convert_to_other_type fail\n";
                });
            tmpLgInfo->p_special_grp = nullptr;
            // global_layers.insert(global_layers.end(), sub_ops.begin(),
            // sub_ops.end()); continue;
          } else {
            tmpLgInfo->_lgInfo.type = GROUP_MM_OPT3;
          }
        }
        tmpLgInfo->base_solver(pass_ir, cycle_calculator_);
        group_cycle += tmpLgInfo->group_cycle;
        sub_ilp_LgInfos.push_back(tmpLgInfo);
      } else {
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::errs() << "ilp_debug: add global_layer:" << show_op_info(grp[0])
                       << "\n";
        });
        global_layers.push_back(grp[0]);
      }
    }
  }
  for (auto global_layer : global_layers) {
    if (global_layer) {
      group_cycle += cycle_calculator_->getGlobalLayerCycle(global_layer);
    }
  }
}

bool ilp_LgInfo::binary_search_group(bool move_right, const LgOptions &options,
                                     std::shared_ptr<dot_graph> dot_graph_log) {
  if (middle_ptr == -1) {
    group_ops_all.assign(_lgInfo.group_ops.begin(), _lgInfo.group_ops.end());
    left_ptr = 0;
    right_ptr = _lgInfo.group_ops.size();
    last_success_middle_ptr = right_ptr / 2;
  } else {
    int tmp_left_ptr, tmp_right_ptr;
    if (move_right) {
      tmp_right_ptr = right_ptr;
      tmp_left_ptr = middle_ptr;
    } else {
      tmp_right_ptr = middle_ptr;
      tmp_left_ptr = left_ptr;
    }
    if (tmp_right_ptr - tmp_left_ptr < 2) {
      int i = 0;
      _lgInfo.group_ops.clear();
      for (auto op : group_ops_all) {
        if (i >= last_success_middle_ptr) {
          divided_group_ops.push_back(op);
        } else {
          _lgInfo.group_ops.push_back(op);
        }
        i++;
      }
      return false;
    }
    if (move_right) {
      left_ptr = middle_ptr;
    } else {
      right_ptr = middle_ptr;
    }
  }
  middle_ptr = left_ptr + (right_ptr - left_ptr) / 2;
  while (true) {
    auto nodes = GetParallelNodes(group_ops_all[middle_ptr]);
    if (!nodes.size()) {
      LAYER_GROUP_LOG_DEBUG_BLOCK(
          { llvm::errs() << "ParallelNodes is None, middle_ptr ok\n"; });
      break;
    }
    if (++middle_ptr >= right_ptr - 1) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::errs() << "inc middle_ptr\n"; });
      break;
    }
  }

  if (move_right && pre_middle_ptr != -1) {
    last_success_middle_ptr = pre_middle_ptr;
  }
  auto cut_op = group_ops_all[middle_ptr];
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::errs() << "ilp_debug: binary_search_group, middle_ptr:" << middle_ptr
                 << ", op(excluded):" << show_op_info(cut_op) << "\n";
  });
  auto name = module::getName(cut_op).str();
  dot_graph_log->add_node_label(name + "_ori",
                                std::string("binary_search_cut_op"));
  int i = 0;
  _lgInfo.group_ops.clear();
  for (auto op : group_ops_all) {
    if (i < middle_ptr) {
      _lgInfo.group_ops.push_back(op);
    }
    i++;
  }
  _lgInfo.update_group_io(options_.opt);
  // set_group_type(_lgInfo);
  _lgInfo.type = GROUP_NORMAL;
  pre_middle_ptr = middle_ptr;
  return true;
}

std::vector<Operation *> ilp_LgInfo::GetParallelNodes(Operation *op) {
  if (!map_parallel_node.size()) {
    GetAllParallelNodes(_lgInfo.group_ops, map_parallel_node,
                        &_lgInfo.group_ops);
  }
  return map_parallel_node[op];
}

static void collectAllSubLgInfoResult(std::shared_ptr<ilp_LgInfo> lgInfo,
                                      LgPassIR *pass_ir) {
  if (lgInfo->group_success) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({
      llvm::errs() << "add group to LgPassIR:" << lgInfo->_lgInfo.group_id
                   << "\n";
    });
    lgInfo->save_result(pass_ir);
  }

  for (auto it : lgInfo->sub_ilp_LgInfos) {
    if (it->group_success) {
      // LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::errs()<<"add group to
      // LgPassIR:"<<it->_lgInfo.group_id<<"\n";}); llvm::errs()<<"add group to
      // LgPassIR:"<<it->_lgInfo.group_id<<"\n";
      it->save_result(pass_ir);
    } else {
      collectAllSubLgInfoResult(it, pass_ir);
    }
  }
}

static void
collectAllSubLgInfo(std::shared_ptr<ilp_LgInfo> lgInfo,
                    std::vector<std::shared_ptr<ilp_LgInfo>> &base_groups) {
  if (lgInfo->group_success) {
    base_groups.push_back(lgInfo);
  }

  for (auto sub_lgInfo : lgInfo->sub_ilp_LgInfos) {
    if (sub_lgInfo->group_success) {
      base_groups.push_back(sub_lgInfo);
    } else {
      collectAllSubLgInfo(sub_lgInfo, base_groups);
    }
  }
}

static std::shared_ptr<std::vector<std::shared_ptr<ilp_LgInfo>>>
expandAllNestedLgInfo(std::vector<std::shared_ptr<ilp_LgInfo>> &base_groups) {
  auto new_base_groups =
      std::make_shared<std::vector<std::shared_ptr<ilp_LgInfo>>>();
  for (int64_t i = 0, grp_num = base_groups.size(); i < grp_num; i++) {
    collectAllSubLgInfo(base_groups[i], *new_base_groups);
  }
  return std::move(new_base_groups);
}

void GroupMethod::ilp_layer_group(LgPassIR *pass_ir) {
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::errs() << "\n"
                 << "=======================================================\n"
                 << "*********** ilp_layer_group **********\n"
                 << "=======================================================\n";
  });
  std::vector<Operation *> subnet_ops;
  Operation *dot_root_op = nullptr;
  for (auto it : pass_ir->subnet_ops) {
    if (!dot_root_op &&
        module::isDebugCmdEnable("dot_root_op_name-" +
                                 module::getName(it).str() + "-")) {
      llvm::errs() << "ilp_layer_group find dot_root_op_name:"
                   << module::getName(it).str() << "\n";
      dot_root_op = it;
    }
    subnet_ops.push_back(it);
  }
  if (module::isDebugCmdEnable("dot_root_op_name") && dot_root_op) {
    std::vector<Operation *> op_tree, exclude_ops, break_ops;
    find_op_tree_by_root2(dot_root_op, op_tree, subnet_ops, exclude_ops,
                          break_ops, 0, 8);
    auto dot_graph_log = createSubnetGraph(op_tree);
    dot_graph_log->export_dot(
        "svg_initial2_" + module::getName(module::getModuleOp()).str(), true);
  }

  pass_ir->dot_graph_log_subnet = createSubnetGraph(subnet_ops);
  //------------------------part0: pre
  // processing----------------------------------------------------
  init_ilp_base_groups(pass_ir);
  pass_ir->dot_graph_log_subnet->add_node_label(
      "global_info",
      "init group_num:" + std::to_string(pass_ir->tmp_base_groups.size()));
  if (module::isDebugCmdEnable("export_full_svg")) {
    pass_ir->dot_graph_log_subnet->export_dot(
        "svg_initial2_" + module::getName(module::getModuleOp()).str(), true);
  }

  //------------------------part1:
  // processing----------------------------------------------------
  std::vector<std::shared_ptr<ilp_LgInfo>> base_groups2;
  bool specify_group = false;
  for (int64_t i = 0, grp_num = pass_ir->tmp_base_groups.size(); i < grp_num;
       i++) {
    if (module::isDebugCmdEnable("save_mlir_file_for_group_id" +
                                 std::to_string(i))) {
      ilp_func_trace tmp_trace(llvm::formatv("high_solver, i:{0}", i).str());
      auto best_lgInfo =
          pass_ir->tmp_base_groups[i]->high_solver(pass_ir, cycle_calculator_);
      base_groups2.push_back(best_lgInfo ? best_lgInfo
                                         : pass_ir->tmp_base_groups[i]);
      specify_group = true;
      break;
    }
  }
  if (!specify_group) {
    for (int64_t i = 0, grp_num = pass_ir->tmp_base_groups.size(); i < grp_num;
         i++) {
      ilp_func_trace tmp_trace(llvm::formatv("high_solver, i:{0}", i).str());
      auto best_lgInfo =
          pass_ir->tmp_base_groups[i]->high_solver(pass_ir, cycle_calculator_);
      base_groups2.push_back(best_lgInfo ? best_lgInfo
                                         : pass_ir->tmp_base_groups[i]);
    }
  }

  auto base_groups3 = expandAllNestedLgInfo(base_groups2);
  if (module::isDebugCmdEnable("opt3_o3")) {
    try_cut_some_group(pass_ir, *base_groups3, true);
  }

  auto base_groups4 = expandAllNestedLgInfo(*base_groups3);
  if (module::isDebugCmdEnable("opt3_o2")) {
    try_cut_some_group(pass_ir, *base_groups4, false);
  }
  // try_modify_mlp_group_sub_sum(pass_ir, *base_groups4, cycle_calculator_);
  auto base_groups5 = expandAllNestedLgInfo(*base_groups4);
  for (int64_t i = 0, grp_num = base_groups5->size(); i < grp_num; i++) {
    collectAllSubLgInfoResult((*base_groups5)[i], pass_ir);
  }

  if (module::isDebugCmdEnable("export_full_svg")) {
    pass_ir->dot_graph_log_subnet->add_node_label(
        "global_info",
        "final group_num:" + std::to_string(pass_ir->lg_infos.size()));
    for (auto [grp_idx, lg_info] : llvm::enumerate(pass_ir->lg_infos)) {
      for (auto [op_idx, op] : llvm::enumerate(lg_info.group_ops)) {
        if (op) {
          pass_ir->dot_graph_log_subnet->add_node_label(
              module::getName(op).str(),
              "grp_" + std::to_string(grp_idx) + "*_id_" +
                  std::to_string(lg_info.group_id) + "*_" +
                  std::to_string(op_idx) + "*");
        }
      }
    }
    pass_ir->dot_graph_log_subnet->export_dot(
        "svg_" + module::getName(module::getModuleOp()).str());
  }
}

} // namespace tpu
} // namespace tpu_mlir

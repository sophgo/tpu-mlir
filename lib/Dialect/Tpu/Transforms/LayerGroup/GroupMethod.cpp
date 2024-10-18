//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Backend/BM168x/BackendInterfaces.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupMethod.h"
#include "progressbar.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include <llvm/Support/Debug.h>
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/IlpTimeStep.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/TimeStepMethod.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgPass.h"
#include <random>

#define DEBUG_TYPE "layer-group"
using namespace tpu_mlir::backend;

namespace tpu_mlir {
namespace tpu {
#define MAX_GROUP_CLUSTER (50)

#define GROUP_CHECK_RETURN(val)                                                \
  {                                                                            \
    if (val) {                                                                 \
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "layer group is valid";});  \
      return true;                                                             \
    } else {                                                                   \
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "layer group is invalid";}); \
      return false;                                                             \
    }                                                                          \
  }

bool opt_cost_all = false;

bool Sort_by_int(const std::pair<Value, int64_t> &v1,
                 const std::pair<Value, int64_t> &v2) {
  return v1.second < v2.second; // ��������
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
    std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister 19937 生成器
    std::uniform_int_distribution<> dis(0, charset.length() - 1);

    std::string s;
    s.reserve(length); // 预分配字符串空间以提高效率
    for (int i = 0; i < length; ++i) {
        s += charset[dis(gen)]; // 从字符集中随机选取字符
    }
    return s;
}

class ilp_func_trace {
  public:
    ilp_func_trace(std::string debug_info, int64_t specified_id = 0, std::shared_ptr<dot_graph> dot_graph_log = nullptr) {
      _debug_info = debug_info;
      _dot_graph_log = dot_graph_log;
      string_id =  specified_id == 0?GenerateRandomString(15):std::to_string(specified_id);
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() <<string_id<<" ilp_debug: "<<_debug_info<<" start\n";});
    }

    ~ilp_func_trace() {
      std::string extra_info = "";
      if (_dot_graph_log) {
        std::string svg_file = _dot_graph_log->export_dot("svg_" + _debug_info +"_" + string_id);
        extra_info = ", please refer svg:" + svg_file;
      }
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() <<string_id<<" ilp_debug: "<<_debug_info<<" end"<<extra_info<<"\n";});
    }
  private:
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
    if(isa<ReshapeOp>(op)){
      auto ishape = module::getShape(op->getOperand(0));
      auto oshape = module::getShape(op->getResult(0));
      if(ishape.size() > 5 || oshape.size() > 5){
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

void GroupMethod::get_layer_group(LgInfo &lg_info,
                            const std::vector<Operation *> &base_group,
                            int64_t left, int64_t right) {
  lg_info.clear();
  for (int idx = left; idx <= right; ++idx) {
    lg_info.group_ops.push_back(base_group[idx]);
  }
  lg_info.update_group_io(opt4_ori_opt_ < 0 ? opt_ : opt4_ori_opt_);
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

void GroupMethod::get_base_branch_groups(
    std::vector<std::shared_ptr<ilp_LgInfo>> &base_groups,
    const llvm::SetVector<Operation *> &subnet_ops,
    const std::vector<Value> &subnet_return_opds) {
  /*std::vector<std::vector<Operation *>> tmp_base_groups;
  for (auto v : subnet_return_opds) { // ���������value��ʼ����
    auto tmp_op = v.getDefiningOp();
    std::vector<Operation *> tmp;
    tmp.push_back(tmp_op);
    tmp_base_groups.push_back(tmp);
  }

  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::outs() << "get_base_branch_groups start, group num:"
                 << tmp_base_groups.size() << "\n";
  });
  while (true) {
    bool can_break = true;
    for (auto group : tmp_base_groups) { // �ж��Ƿ����к�ѡ���Ѵ������
      if (group.back() != nullptr) {
        can_break = false;
        break;
      }
    }
    if (can_break) {
      break; // �Ѵ�����ϣ��˳�ѭ��
    }

    for (auto &group : tmp_base_groups) {
      auto tmp_op = group.back();
      if (tmp_op == nullptr) {
        continue;
      }
      int count = 0, imm_tensor_idx = 0, idx = 0;
      for (auto v : tmp_op->getOperands()) {
        auto pre_op = v.getDefiningOp();
        if (pre_op == nullptr ||
            isa<top::NoneOp, top::WeightOp, top::InputOp>(pre_op)) {
          continue;
        }
        if (isPreOpHaveAComputeOp(pre_op)) {
          count++;
          imm_tensor_idx = idx;
        }
        idx++;
      }
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "op:" << module::getName(tmp_op).str() << " have "
                   << count << " input tensor is not weight\n";});
      if (count == 1) {
        auto tmp_op2 = tmp_op->getOperand(imm_tensor_idx).getDefiningOp();
        int user_count = 0;
        for (auto itr : tmp_op2->getResult(0).getUsers()) {
          if (!isa<ReturnOp>(itr)) {
            user_count++;
          }
        }
        LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "have " << user_count << " next node\n";});
        if (user_count > 1) { // �����ֲ��
          group.push_back(nullptr);
          bool grp_exist = false;
          for (auto tmp_group : tmp_base_groups) {
            if (tmp_op2 == tmp_group[0]) {
              grp_exist = true;
              break;
            }
          }
          if (!grp_exist) {
            LAYER_GROUP_LOG_DEBUG_BLOCK({
              llvm::outs() << "meet divide node, add new group, start op name:"
                           << module::getName(tmp_op2).str() << "\n";
            });
            std::vector<Operation *> tmp;
            tmp.push_back(tmp_op2);
            tmp_base_groups.push_back(tmp);
          }
        } else {
          group.push_back(tmp_op2);
        }
      } else if (count > 1) {     // ������ϵ�
        group.push_back(nullptr); // ������ǰ��ѡ��
        for (auto v : tmp_op->getOperands()) { // ����ϵ�����2���·�֧��Ϊ�µĺ�ѡ��
          auto pre_op = v.getDefiningOp();
          if (pre_op != nullptr &&
              isa<top::NoneOp, top::WeightOp, top::InputOp>(pre_op)) {
            continue;
          }
          LAYER_GROUP_LOG_DEBUG_BLOCK({
            llvm::outs() << "meet merge node, add new group, start op name:"
                         << module::getName(pre_op).str() << "\n";
          });
          std::vector<Operation *> tmp;
          tmp.push_back(pre_op);
          tmp_base_groups.push_back(tmp);
        }
        break; // �������base_groups����Ԫ�أ���������ѭ��������Ӱ��ѭ������
      } else {
        group.push_back(nullptr);
        break; // �����������
      }
    }
  }

  for (auto &group : tmp_base_groups) {
    group.pop_back();
    reverse(group.begin(), group.end());
  }

  int i = 0;
  for (auto group : tmp_base_groups) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << ">>>tmp_base_groups grp:" << i++ << "\n";});
    int j = 0;
    for (auto op : group) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "  op:" << j++ << " name: " << module::getName(op).str()
                   << "\n";});
    }
  }

  base_groups.clear();
  for (auto group : tmp_base_groups) {
    std::vector<Operation *> tmp;
    for (auto op : group) {
      if (isLgSupport(op)) {
        tmp.push_back(op);
      } else {
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::outs() << "global layer name: " << module::getName(op).str()
                       << "\n";
        });
        if (tmp.size() > 1) {
          base_groups.push_back(tmp);
        }
        tmp.clear();
      }
    }
    if (tmp.size() > 1) {
      base_groups.push_back(tmp);
    }
  }

  i = 0;
  for (auto group : base_groups) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << ">>>base_groups grp:" << i++ << "\n";});
    int j = 0;
    for (auto op : group) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "  op:" << j++ << " name: " << module::getName(op).str()
                   << "\n";});
    }
  }*/
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

void GroupMethod::get_base_dfs_topo_groups(std::vector<std::shared_ptr<ilp_LgInfo>> &tmp_base_groups) {
  int idx = 0;
  for (auto& grp : tmp_base_groups) {
    auto& ops = grp->_lgInfo.group_ops;
    idx++;
    if (ops.size() == 1) {
      continue;
    }
    llvm::errs() << "start refine order, grp:" << --idx << "\n";
    std::vector<Operation*> topo_ops;
    std::map<Operation*, int> indeg;
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
      llvm::errs() << "  op:" << i++ << ": " <<show_op_info(op)<< "\n";
    }
    ops.assign(topo_ops.begin(), topo_ops.end());
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
  PROFILE_LOG("is_layer_group_valid", true);
  bool status;
  status = group_one_layer_proc(lg_info, calc_cost, group_cost);
  if (status && LgPass::OPTIONS.group_by_cores == false) {
    PROFILE_LOG("is_layer_group_valid", false);
    return true;
  }

  if (!group_valid_pre_check(lg_info)) {
    PROFILE_LOG("is_layer_group_valid", false);
    return false;
  }

  shape_secs_t shape_secs;
  std::vector<std::pair<Value, int64_t>> value_size;
  if (!init_group_data_secs(lg_info, shape_secs, value_size)) {
    PROFILE_LOG("is_layer_group_valid", false);
    return false;
  }
  DEBUG_WITH_TYPE("shape_secs", {
    llvm::dbgs() << "; action = init_group_data_secs" <<
    "; nsecs = " << shape_secs.nsecs <<
    "; csecs = " << shape_secs.csecs <<
    "; dsecs = " << shape_secs.dsecs <<
    "; hsecs = " << shape_secs.hsecs <<
    "; wsecs = " << shape_secs.wsecs <<
     "\n";
  });
  if (!dynamic_group_valid_check(lg_info)) {
    PROFILE_LOG("is_layer_group_valid", false);
    return false;
  }

  auto time_step = std::make_shared<BasicTimeStep>();
  status = time_step->assignTimeStep(lg_info, shape_secs, true);
  if (status == false) {
    PROFILE_LOG("is_layer_group_valid", false);
    return false;
  }

  auto lmem_allocator = std::make_shared<LmemAllocator>();
  status =
      lmem_allocator->assignLmemAddrWithSecs(lg_info, time_step, shape_secs);
  if (status == false) {
    PROFILE_LOG("is_layer_group_valid", false);
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
  LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "clusters idx(size): ";});
  for (size_t i = 0; i < clusters.size(); ++i) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << llvm::format("%d(%d), ", clusters[i].first,
                                 clusters[i].second);});
  }
  LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "\n";});
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

static std::string format_op_in_out_info(Operation* op) {
  std::string tmpStr = " ";
  int64_t n, c, d, h, w;
  for (auto [index, in] : llvm::enumerate(get_input_values(op))) {
    if (is_value_weight(in)) {
      module::getNCDHW(in, n, c, d, h, w, GROUP_NORMAL);
      tmpStr = tmpStr + llvm::formatv(" in{0}:[{1},{2},{3},{4},{5}]",
              index, n, c, d, h, w).str();

    }
  }
  tmpStr = tmpStr + ", ";
  auto outs = get_output_values(op);
  module::getNCDHW(outs[0], n, c, d, h, w, GROUP_NORMAL);
  tmpStr = tmpStr + llvm::formatv(" out:[{1},{2},{3},{4},{5}], num:{6}",
          index, n, c, d, h, w, outs.size()).str();
  return tmpStr;
}

std::shared_ptr<dot_graph> createSubnetGraph(std::vector<Operation*>& ops) {
  std::shared_ptr<dot_graph> dot_graph_log = std::make_shared<dot_graph>();
  for (auto op : ops) {
    if (!isa<ReturnOp>(op)) {
      auto op_name = module::getName(op).str();
      dot_graph_log->add_node_into_graph(op_name);
      dot_graph_log->add_node_label(op_name,
        op->getName().getStringRef().str() + format_op_in_out_info(op));
      bool next_layer_has_return = false;
      for (auto itr = op->user_begin(); itr != op->user_end(); itr++) {
        if (!isa<ReturnOp>(*itr)) {
          auto to = module::getName(*itr).str();
          dot_graph_log->add_node_into_graph(to);
          dot_graph_log->add_node_label(to,
            (*itr)->getName().getStringRef().str() + format_op_in_out_info(*itr));
          dot_graph_log->add_edge_into_graph(op_name, to);
        } else {
          next_layer_has_return = true;
        }
      }
      if (next_layer_has_return) {
        dot_graph_log->add_node_label(op_name, std::string("to_returnOp"));
      }
    }
  }
  return dot_graph_log;
}

void GroupMethod::dynamic_programming_layer_group_with_cluster(
    std::vector<LgInfo> &lg_infos, const llvm::SetVector<Operation *> &subnet_ops) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({  llvm::outs() << "\n"
               << "=======================================================\n"
               << "***** Dynamic Programming layer group with cluster ****\n"
               << "=======================================================\n";});
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
  LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << llvm::format("total num of base_group is %d\n",
                               base_groups.size());});
  for (size_t i = 0; i < base_groups.size(); ++i) {
    std::vector<std::pair<int64_t, int64_t>> clusters;
    get_group_clusters(clusters, base_groups[i]);
    size_t cluster_num = clusters.size();
    LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << llvm::format(
        "process base group %d, layer_num=%d, cluster_num=%d\n", i,
        base_groups[i].size(), cluster_num);});
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

        DEBUG_WITH_TYPE("lg_cost",{
            llvm::errs()
                        << "; action = lg_cost"
                        << "; step = global_layer"
                        << "; start_idx = " << start_idx
                         << "; end_idx = " << end_idx
                         << "; group_idx = " << i
                         << "; group_cost = " << cost_table[j][j] << "\n";
        });


        cut_points[j][j] = j;
      }

      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "Searching best group slices...\n";});
      progressbar bar(cluster_num - 1);

      /**
       * you can debug any cluster like calc_cost(16, 17);
       */
      auto calc_cost = [&](int64_t start_idx, int64_t end_idx) {
        get_layer_group(sub_group, base_groups[i], start_idx, end_idx);
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
                            << "; start_idx = " << start
                            << "; end_idx = " << end
                            << "; sweep = " << sweep
                            << "; group_idx = " << i
                            << "; group_cost = " << group_cost << "\n";
              });
            }
          }

          DEBUG_WITH_TYPE("lg_cost", {
            llvm::dbgs() << "; action = lg_cost"
                         << "; step = update_better"
                         << "; start_idx = " << start_idx
                         << "; end_idx = " << end_idx
                         << "; group_idx = " << i
                         << "; group_cost = " << group_cost << "\n";
          });

          cost_table[start][end] = group_cost;
          cut_points[start][end] = optimal_point;
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
          get_layer_group(sub_group, base_groups[i], start, end);
          int64_t group_cost = MAX_COST;
          auto temp_status = is_layer_group_valid(sub_group, true, &group_cost);
          llvm::dbgs() << temp_status << " ;start" << start << " - " << " end "
                       << end << " = " << group_cost << "\n";
          start = end + 1;
        }

        llvm::dbgs() << "\n";
        llvm::dbgs() << "================FINAL GROUP================\n";
        for (size_t cost_i = 0; cost_i < cluster_num; ++cost_i) {
          for (int64_t cost_j = 0; cost_j < cluster_num; ++cost_j) {
            llvm::dbgs() << cut_points[cost_i][cost_j] << ", " << "";
          }
          llvm::dbgs() << "\n";
        }
        llvm::dbgs() << "================COST TABLE================\n";
        for (size_t cost_i = 0; cost_i < cluster_num; ++cost_i) {
          for (int64_t cost_j = 0; cost_j < cluster_num; ++cost_j) {
            llvm::dbgs() << cost_table[cost_i][cost_j] << ", " << "";
          }
          llvm::dbgs() << "\n";
        }
        llvm::dbgs() << "=============================================\n";
        llvm::dbgs() << "\n";
      });
    } else {
      cut_results_.push_back(std::vector<int64_t>(1, 0));
      DEBUG_WITH_TYPE("lg_cost", {
        if (!isa<ReturnOp>(base_groups[i][0]) && runmode_ == RunMode::TPU_STATIC) {
          int64_t start_idx = clusters[0].first;
          // int64_t end_idx = start_idx + clusters[0].second - 1;
          get_layer_group(sub_group, base_groups[i], start_idx, start_idx);
          int64_t cost;

          assert(is_layer_group_valid(sub_group, true, &cost));
          llvm::dbgs()  << "; action = lg_cost"
                        << "; step = global_layer"
                        << "; start_idx = " << 0 << "; end_idx = " << 0
                        << "; group_idx = " << i << "; group_cost = " << cost
                       << "\n";
        }else{
          llvm::dbgs() << "; action = lg_cost"
                       << "; step = global_layer"
                       << "; start_idx = " << 0 << "; end_idx = " << 0
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
  // std::vector<int64_t> override_is = {8, 10, 12, 16, 24, 26, 36, 42, 77, 91, 126, 133};
  // std::vector<int64_t> override_is = {8, 10, 12, 20, 22, 24, 26, 32, 34, 36, 44, 45, 46, 47, 54, 58, 74, 76, 78, 90, 98, 105, 107, 109, 112, 119, 120, 121, 122, 126, 133};
  // cut_results_[0] = override_is;
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
    if (!init_group_data_secs(*groups[i], shape_secs[i], value_size)) {
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
    group_costs[i] = cycle_calculator_->getGroupCycle(
        time_steps[i], shape_secs[i], groups[i]->type);
  }
  if (!valid) {
    return false;
  }
  total_cost = group_costs[0] + group_costs[1];
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
            LAYER_GROUP_LOG_DEBUG_BLOCK({
              llvm::outs() << "//// Group cost " << seq_info.min_cost
                           << ", optimal cut idx " << optimal_cut_idx << "\n";
            });
          }
        }
        cut_result[j] = optimal_cut_idx;
        LLVM_DEBUG({
          llvm::dbgs() << "; start_idx = " << left_cut_idx
                        << "; end_idx = " << cut_result[j]
                        << "; group_cost = " << seq_info.left_cost
                        << "; group_idx = " << i
                        << "; action = " << "consider_redundant_computation_and_gdma_cost"
                        << "\n";
          llvm::dbgs() << "; start_idx = " << cut_result[j] + 1
                        << "; end_idx = " << cut_result[j + 1]
                        << "; group_cost = " << seq_info.right_cost
                        << "; group_idx = " << i
                        << "; action = " << "consider_redundant_computation_and_gdma_cost"
                        << "\n";
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
            LLVM_DEBUG({
              llvm::dbgs() << "; start_idx = " << start_cut_idx
                           << "; end_idx = " << end_cut_idx
                           << "; group_cost = " << combine_group_cost
                           << "; group_idx = " << i
                           << "; action = " << "merge_cut_idx_to_reduce_gdma_cost"
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

void GroupMethod::simple_layer_group(std::vector<LgInfo> &lg_infos,
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

static inline int64_t increase_nsecs2(int64_t nsecs, int64_t batch_size) {
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

static inline void update_shape_secs2(const LgInfo &lg_info,
                                      shape_secs_t &shape_secs,
                                      int64_t &dhw_secs,
                                      const shape_secs_t &max_shape_secs) {
  if (shape_secs.nsecs < max_shape_secs.nsecs) {
    shape_secs.nsecs = increase_nsecs2(shape_secs.nsecs, max_shape_secs.nsecs);
  } else if (shape_secs.csecs < max_shape_secs.csecs) {
    shape_secs.csecs = increase_csecs2(shape_secs.csecs, max_shape_secs.csecs);
  } else {
    assign_dhwsecs(lg_info, shape_secs, ++dhw_secs, max_shape_secs);
  }
}

static inline bool update_shape_secs_for_ilp_group(shape_secs_t &shape_secs,const shape_secs_t &max_shape_secs) {

  //
  // if (shape_secs.nsecs < max_shape_secs.nsecs) {
  //   int64_t nsecs = shape_secs.nsecs;
  //   int64_t batch_size = max_shape_secs.nsecs;
  //   int64_t nslice = batch_size / nsecs + (batch_size % nsecs > 0);
  //   int64_t new_nslice = nslice;
  //   int64_t next_nsecs = nsecs;

  //   do {
  //     next_nsecs++;
  //     new_nslice = batch_size / next_nsecs + (batch_size % next_nsecs > 0);
  //   } while (new_nslice >= nslice && next_nsecs < batch_size);

  //   shape_secs.nsecs = next_nsecs;
  // } else if (shape_secs.hsecs < max_shape_secs.hsecs) {
  //   int64_t hsecs = shape_secs.hsecs;
  //   int64_t max_hsecs = max_shape_secs.hsecs;
  //   int64_t hslice = max_hsecs / hsecs + (max_hsecs % hsecs > 0);
  //   int64_t new_hslice = hslice;
  //   int64_t next_hsecs = hsecs;

  //   do {
  //     next_hsecs++;
  //     new_hslice = max_hsecs / next_hsecs + (max_hsecs % next_hsecs > 0);
  //   } while (new_hslice >= hslice && next_hsecs < max_hsecs);

  //     shape_secs.hsecs = next_hsecs;
  //   }

  //
  bool updated = false;
  if (shape_secs.nsecs < max_shape_secs.nsecs) {
    shape_secs.nsecs++;
    updated = true;
  } else if (shape_secs.hsecs < max_shape_secs.hsecs) {
    shape_secs.hsecs++;
    updated = true;
  }

  return updated;
}

std::vector<int>
get_sec_per_cores(const shape_secs_t &shape_secs,
                               std::vector<std::vector<int64_t>> &vec_ncdhw,
                               int core_num, TensorInfo &tensor_infos) {
  int secs = shape_secs.nsecs * shape_secs.csecs * shape_secs.dsecs *
             shape_secs.hsecs * shape_secs.wsecs;
  int secs_per_core = secs / core_num;
  auto sec_per_cores = std::vector<int>();
  for (int i = 0; i < core_num; i++) {
    sec_per_cores.push_back(secs_per_core);
  }
  int rest = secs - core_num * secs_per_core;
  for (int i = 0; i < core_num; i++) {
    if (--rest < 0) {
      break;
    }
    sec_per_cores[i]++;
  }

  for (int i = 0; i < core_num; i++) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "sec_per_cores:" << sec_per_cores[i] << "\n";});
  }

  for (int n = 0; n < shape_secs.nsecs; n++) { // todo Ѱ���ø���core����ˮһ�µ�˳��
    for (int c = 0; c < shape_secs.csecs; c++) {
      for (int d = 0; d < shape_secs.dsecs; d++) {
        for (int h = 0; h < shape_secs.hsecs; h++) {
          for (int w = 0; w < shape_secs.wsecs; w++) {
            std::vector<int64_t> tmp;
            tmp.push_back(n);
            tmp.push_back(c);
            tmp.push_back(d);
            tmp.push_back(h);
            tmp.push_back(w);
            vec_ncdhw.push_back(tmp);
          }
        }
      }
    }
  }

  // struct vec_ncdhw_compare {
  //   bool operator()(std::vector<int> v0, std::vector<int> v1) const {
  //     for (int i = 0; i < v0.size(); i++) {
  //       if (v0[i] < v1[i]) {
  //         return true;
  //       }
  //     }
  //     return false;
  //   }
  // };

  // std::set<std::vector<int>, vec_ncdhw_compare> unique_ncdhw_set;
  // for (auto itr = tensor_infos.begin(); itr != tensor_infos.end(); ++itr) {
  //   if (itr->second.slice_info.n[itr1[0]] !=
  //   itr->second.slice_info.n[itr2[0]] ||
  //       itr->second.slice_info.c[itr1[1]] !=
  //       itr->second.slice_info.c[itr2[1]] ||
  //       itr->second.slice_info.d[itr1[2]] !=
  //       itr->second.slice_info.d[itr2[2]] ||
  //       itr->second.slice_info.h[itr1[3]] !=
  //       itr->second.slice_info.h[itr2[3]] ||
  //       itr->second.slice_info.w[itr1[4]] !=
  //       itr->second.slice_info.w[itr2[4]]) {
  //         return false;
  //   }
  // }

  // for (auto itr1 = vec_ncdhw.begin(); itr1 != vec_ncdhw.end(); ++itr) {
  //   for (auto itr2 = vec_ncdhw.begin(); itr2 != vec_ncdhw.end(); ++itr2) {
  //     for (auto itr = tensor_infos.begin(); itr != tensor_infos.end(); ++itr)
  //     {
  //       if (itr->second.slice_info.n[itr1[0]] !=
  //       itr->second.slice_info.n[itr2[0]] ||
  //           itr->second.slice_info.c[itr1[1]] !=
  //           itr->second.slice_info.c[itr2[1]] ||
  //           itr->second.slice_info.d[itr1[2]] !=
  //           itr->second.slice_info.d[itr2[2]] ||
  //           itr->second.slice_info.h[itr1[3]] !=
  //           itr->second.slice_info.h[itr2[3]] ||
  //           itr->second.slice_info.w[itr1[4]] !=
  //           itr->second.slice_info.w[itr2[4]]) {

  //       }
  //     }
  //   }
  // }

  // reverse(vec_ncdhw.begin(),vec_ncdhw.end());
  return std::move(sec_per_cores);
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
  LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "old overlap:" << overlap << "\n";});
  if (op_num <= overlap) {
    overlap = 1;
  } else if (op_num * 0.2 > overlap) {
    overlap = op_num * 0.2;
  }
  LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "new overlap:" << overlap << "\n";});
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


static void find_op_in_same_block(Operation *op,
                                  std::vector<Operation *> &group_ops,
                                  std::map<Operation *, int> &op_block_id,
                                  int in_idx) {
  if (std::find(group_ops.begin(), group_ops.end(), op) == group_ops.end()) {
    return;
  }
  if (op_block_id.find(op) != op_block_id.end()) {
    return;
  }
  op_block_id[op] = in_idx;
  for (auto v : op->getOperands()) {
    auto pre_op = v.getDefiningOp();
    if (pre_op == nullptr ||
        isa<top::NoneOp, top::WeightOp, top::InputOp>(pre_op)) {
      continue;
    }
    if (std::find(group_ops.begin(), group_ops.end(), pre_op) !=
        group_ops.end()) {
      find_op_in_same_block(pre_op, group_ops, op_block_id, in_idx);
    }
  }

  for (auto user : op->getUsers()) {
    if (std::find(group_ops.begin(), group_ops.end(), user) !=
        group_ops.end()) {
      find_op_in_same_block(user, group_ops, op_block_id, in_idx);
    }
  }
}


void GroupMethod::cut_this_group_is_better(ilp_LgInfo& original_group, LgPassIR *pass_ir,
                                          std::vector<std::shared_ptr<ilp_LgInfo>>& base_groups)
{
  std::vector<Operation*> cut_ops;
  auto group_ops = original_group._lgInfo.group_ops;
  for(int i=0; i < group_ops.size(); i++) {
    auto cur_op = group_ops[i];
    if(isa<tpu::Conv2DOp, tpu::DeconvOp>(cur_op)) {
      cut_ops.push_back(cur_op);
    }
  }
  if (cut_ops.size()==0) {
    original_group.conv_cut_optimized = true;
    return;
  }

  Operation* best_cut_op = nullptr;
  int64_t min_group_cost = 10000000000, idx = 0;
  for (auto cut_op: cut_ops) {
    if (opHasMultiGroupUser(cut_op, group_ops)) {
      continue;
    }
    LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() <<"ilp_debug, idx:"<<idx++<< ", process_cut_op:"<< show_op_info(cut_op) << "\n";});
    std::vector<Operation*> tmp_ops;
    tmp_ops.assign(group_ops.begin(), group_ops.end());
    std::vector<Operation *> right_sub_ops;
    find_all_next_ops(cut_op, right_sub_ops, &group_ops);
    for (auto right_sub_op : right_sub_ops) {
      tmp_ops.erase(std::remove(tmp_ops.begin(), tmp_ops.end(), right_sub_op),
                      tmp_ops.end());
    }
    right_sub_ops.erase(std::remove(right_sub_ops.begin(), right_sub_ops.end(), cut_op),
                    right_sub_ops.end());

    uint64_t left_sub_group_cost = std::numeric_limits<uint64_t>::max();
    uint64_t right_sub_group_cost = std::numeric_limits<uint64_t>::max();
    if (tmp_ops.size() > 0) {
      if (tmp_ops.size()==1) {
        left_sub_group_cost = cycle_calculator_->getGlobalLayerCycle(tmp_ops.back());
      } else {
        auto left_sub_group = CreateIlpLgInfo(sortOpsByOtherOpsOrder(group_ops, tmp_ops), STRATEGY_SEARCH_CONV_CUT);
        left_sub_group->base_solver(pass_ir, cycle_calculator_);
        if (left_sub_group->group_cycle > 0) { //确保group成功才行
          left_sub_group_cost = left_sub_group->group_cycle;
        }
      }
    } else {
      left_sub_group_cost = 0;
    }

    if (right_sub_ops.size() > 0) {
      if (right_sub_ops.size()==1) {
        right_sub_group_cost = cycle_calculator_->getGlobalLayerCycle(right_sub_ops.back());
      } else {
        auto right_sub_group = CreateIlpLgInfo(sortOpsByOtherOpsOrder(group_ops, right_sub_ops), STRATEGY_SEARCH_CONV_CUT);
        right_sub_group->base_solver(pass_ir, cycle_calculator_);
        if (right_sub_group->group_cycle > 0) {
          right_sub_group_cost = right_sub_group->group_cycle;
        }
      }
    } else {
      right_sub_group_cost = 0;
    }

    int64_t global_op_cost = cycle_calculator_->getGlobalLayerCycle(cut_op);
    int64_t cut_group_cost = left_sub_group_cost + right_sub_group_cost + global_op_cost;
    if (cut_group_cost < original_group.group_cycle && cut_group_cost < min_group_cost) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs()<<"ilp_debug, find_cut_op, original_group.group_cycle:"<< original_group.group_cycle
                  <<", cut_group_cost:"<< cut_group_cost <<", global_op_cost:"<< global_op_cost
                  <<", left_sub_group_cost:"<< left_sub_group_cost
                  <<", right_sub_group_cost:"<< right_sub_group_cost << "\n";});
      best_cut_op = cut_op;
      min_group_cost = cut_group_cost;
    }
  }

  if (best_cut_op) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() <<"ilp_debug, best_cut_op:"<< module::getName(best_cut_op).str() << "\n";});
    std::vector<Operation*> tmp_ops;
    tmp_ops.assign(group_ops.begin(), group_ops.end());
    std::vector<Operation *> right_sub_ops;
    find_all_next_ops(best_cut_op, right_sub_ops, &group_ops);
    for (auto right_sub_op : right_sub_ops) {
      tmp_ops.erase(std::remove(tmp_ops.begin(), tmp_ops.end(), right_sub_op),
                      tmp_ops.end());
    }
    right_sub_ops.erase(std::remove(right_sub_ops.begin(), right_sub_ops.end(), best_cut_op),
                    right_sub_ops.end());

    if (tmp_ops.size() > 1) {
      auto left_sub_group = CreateIlpLgInfo(sortOpsByOtherOpsOrder(group_ops, tmp_ops));
      left_sub_group->base_solver(pass_ir, cycle_calculator_);
      original_group.sub_ilp_LgInfos.push_back(left_sub_group);
    }
    if (right_sub_ops.size() > 1) {
      auto right_sub_group = CreateIlpLgInfo(sortOpsByOtherOpsOrder(group_ops, right_sub_ops));
      right_sub_group->base_solver(pass_ir, cycle_calculator_);
      original_group.sub_ilp_LgInfos.push_back(right_sub_group);
    }
    original_group.group_success = false;
  }

  original_group.conv_cut_optimized = true;
  return;
}

void GroupMethod::try_cut_some_group(LgPassIR *pass_ir, std::vector<std::shared_ptr<ilp_LgInfo>>& base_groups) {
  if (module::isDebugCmdEnable("disable_group_cut")) {
    return;
  }

  ilp_func_trace tmp_trace(__func__);

  while(true) {
    bool all_optimized = true;
    int grp_num = base_groups.size();
    for (int64_t i = 0; i < grp_num; i++) {
      if (!base_groups[i]->conv_cut_optimized && base_groups[i]->_lgInfo.group_ops.size() > 2) {
        ilp_func_trace tmp_trace(llvm::formatv("cut_this_group_is_better, i:{0}", i).str(),
                                 base_groups[i]->_lgInfo.group_id);
        cut_this_group_is_better(*base_groups[i], pass_ir, base_groups);
        all_optimized = false;
      }
    }
    if (all_optimized) {
      break;
    }
  }
}

Operation* check_single_group_could_be_load(LgInfo &sub_group)
{

  std::vector<std::pair<Operation *, int>> vec_op_hwsecs;
  get_group_max_secs(sub_group, vec_op_hwsecs);
  std::sort(vec_op_hwsecs.begin(), vec_op_hwsecs.end(),pair_op_int_Sort_by_int);

  shape_secs_t shape_secs;
  std::vector<std::pair<Value, int64_t>> value_size;

  // 判断切分后内存是否能加载
  if (!init_group_data_secs(sub_group, shape_secs, value_size)) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "init_group_data_secs fail\n";});
      return vec_op_hwsecs[0].first;
    }

  // //考虑反推shape是否正常
  TensorInfo tensor_infos;
  Operation *fail_op = nullptr;
  if (stripe_mine_idx_slice2(sub_group, shape_secs, tensor_infos, fail_op) == false) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({
      llvm::outs() << "stripe_mine_idx_slice2 fail, remove fail_op: "
                   << module::getName(fail_op).str() << "\n";
    });
    return fail_op;
    }

  return nullptr;
}

static void l2m_process(ilp_LgInfo &sub_group, std::vector<std::pair<Value, int64_t>>& value_size) {
  LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "process l2m...\n";});
  auto& grp_time_step = sub_group.timeStepPtrs;
  auto& map_l2m_load = sub_group.map_l2m_load;
  int ts_count = grp_time_step[0]->ts_count;
  int core_num_per_pipe0 = grp_time_step[0]->ncdhw_steps.size();
  for (auto itr : grp_time_step[0]->vec_l2m_value_info) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({
      llvm::outs() << "check Value:" << module::getName(itr.value).str()
                   << ", slice_idx:" << itr.slice_idx
                   << ", pipe0 load ts:" << itr.load_ts << "\n";
    });
    int parallel_core_num = core_num_per_pipe0;
    int min = itr.load_ts;
    for (int j = 1; j < grp_time_step.size(); j++) { //遍历除第1个流水外的其他流水，第1个流水最长
      parallel_core_num += grp_time_step[j]->ncdhw_steps.size();
      for (auto itr3 = grp_time_step[j]->vec_l2m_value_info.begin();
                itr3 != grp_time_step[j]->vec_l2m_value_info.end(); ++itr3) {
        if (itr3->value == itr.value && itr3->slice_idx == itr.slice_idx) {
          LAYER_GROUP_LOG_DEBUG_BLOCK({
            llvm::outs() << "find in pipe:" << j
                         << ", load ts:" << itr3->load_ts << "\n";
          });
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
      for (auto itr: map_l2m_load[m]) {
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::outs() << " Value:" << module::getName(itr.value).str()
                       << " slice_idx:" << itr.slice_idx << " load ts:" << m
                       << " free ts:" << itr.free_ts << "\n";
        });
      }
    }
  }

  int total_weight_size = 0, l2_mem_size = 128*1024*1024;
  int weight_num = value_size.size();
  for (auto it2: value_size) {
    total_weight_size += it2.second;
  }
  l2mem_alloc_Ptr l2mem_alloc_ptr = std::make_shared<l2mem_alloc>();
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
      for (auto it2: value_size) {
        total_weight_size += it2.second;
        if (total_weight_size > l2_mem_size - (int)(share_mem_size*1.5)) {
          break;
        }
        value_size_l2m.push_back(it2);
        value_l2m.push_back(it2.first);
        value_l2m_addr.push_back(addr);
        addr +=it2.second;
      }
      l2mem_alloc_ptr->clear();
      for (auto it3: value_size_l2m) {
        auto name = module::getName(it3.first).str();
        l2mem_alloc_ptr->alloc(-1, name, it3.first, it3.second);
      }

      std::map<int, std::vector<l2m_value_info>> map_l2m_free;
      bool failed = false;
      for (int m = -1; m < ts_count; m++) {
        //处理在该时隙需要释放的l2m tensor
        if (map_l2m_free.find(m) != map_l2m_free.end()) {
          for (auto it3:map_l2m_free[m]) {
            auto name = module::getName(it3.value).str();
            l2mem_alloc_ptr->free(it3.slice_idx, name);
          }
        }
        //处理在该时隙需要分配的l2m tensor
        if (map_l2m_load.find(m) != map_l2m_load.end()) {
          for (auto it3:map_l2m_load[m]) {
            if (std::find(value_l2m.begin(), value_l2m.end(), it3.value) == value_l2m.end()) {
              auto name = module::getName(it3.value).str();
              failed = l2mem_alloc_ptr->alloc(it3.slice_idx, name, it3.value, it3.size);
              if (failed) {
                break;
              }
              //记录当前分配的l2m tensor待释放的时隙
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
    LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "l2m enough \n";});
    for (auto it3: value_size) {
      value_l2m.push_back(it3.first);
      auto name = module::getName(it3.first).str();
      l2mem_alloc_ptr->alloc(-1, name, it3.first, it3.second);
    }
  }

  for (int m = -1; m < ts_count; m++) {
    if (map_l2m_load.find(m) != map_l2m_load.end()) {
      for (auto& itr: map_l2m_load[m]) {
        if (itr.slice_idx > 0 && std::find(value_l2m.begin(), value_l2m.end(), itr.value) != value_l2m.end()) {
          LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "value:" << module::getName(itr.value).str() << ",set valid false\n";});
          itr.valid = false;
        }
      }
    }
  }
  // pass_ir->map_l2m_loads.push_back(map_l2m_load);
  // pass_ir->lg_l2mem_alloc_ptr.push_back(l2mem_alloc_ptr);
  sub_group.l2mem_alloc = l2mem_alloc_ptr;
}

static bool is_same_pipeline(int core_id, std::vector<ILPTimeStepPtr>& timeStepPtrs, int& vec_ncdhw_idx,
                                  TensorInfo& tensor_infos, LgInfo &sub_group,
                                  std::vector<std::vector<int64_t>> vec_ncdhw, std::vector<int>& sec_per_cores) {
  bool all_slice_same = false;
  for (int n = 0; n < timeStepPtrs.size(); n++) { // 遍历历史流水线
    std::vector<std::vector<int64_t>> &ncdhw_steps = timeStepPtrs[n]->ncdhw_steps.begin()->second;
    if (ncdhw_steps.size() == sec_per_cores[core_id]) { // 历史流水线与当前有相同slice数量
      all_slice_same = true;
      for (int m = 0; m < sec_per_cores[core_id]; m++) {
        std::vector<int64_t>& his_steps = ncdhw_steps[m];
        std::vector<int64_t> ncdhw = vec_ncdhw[vec_ncdhw_idx + m];
        slice_info_t &slice_info = tensor_infos[sub_group.group_ops[0]->getOperand(0)] .slice_info; // todo 这里是使用于单分支
        // for (auto itr = tensor_infos.begin(); itr != tensor_infos.end(); ++itr) {
        if (slice_info.n[his_steps[0]].second != slice_info.n[ncdhw[0]].second ||
            slice_info.c[his_steps[1]].second != slice_info.c[ncdhw[1]].second ||
            slice_info.d[his_steps[2]].second != slice_info.d[ncdhw[2]].second ||
            slice_info.h[his_steps[3]].second != slice_info.h[ncdhw[3]].second ||
            slice_info.w[his_steps[4]].second != slice_info.w[ncdhw[4]].second) {
            all_slice_same = false;
            break;
        }
      }
      if (all_slice_same) {
        LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "core " << core_id << ",all slice shape same with pipeline " << n << ", skip ILP\n";});
        for (int m = 0; m < sec_per_cores[core_id]; m++) {
          std::vector<int64_t> ncdhw = vec_ncdhw[vec_ncdhw_idx + m];
          timeStepPtrs[n]->addSliceNcdhwSteps(core_id, ncdhw);
        }
        vec_ncdhw_idx += sec_per_cores[core_id];
        break;
      }
    }
  }
  return all_slice_same?true:false;
}


template <typename opTy>
static bool isOpTypeInGroup(const std::vector<Operation *>& group_ops, std::vector<Operation *>& query_ops) {
  query_ops.clear();
  for (auto op : group_ops) {
    if (isa<opTy>(op)) {
      query_ops.push_back(op);
    }
  }
  return query_ops.size()?true:false;
}


static bool ilp_for_single_group(LgPassIR *pass_ir, ilp_LgInfo &sub_group, int& fail_process_mode,
                                Operation*& fail_op, bool& is_fail_op_in_grp, std::shared_ptr<CycleCalculator> cycle_calculator_) {
  auto tmp_dot_graph_log = pass_ir->dot_graph_log_subnet->clone();
  for (auto [index, op] : llvm::enumerate(sub_group._lgInfo.group_ops)) {
    tmp_dot_graph_log->add_node_label(module::getName(op).str(),
        "grp_ts" + std::to_string(index + 1) +"*");
  }

  bool ret = false;
  ilp_func_trace tmp_trace(__func__, sub_group._lgInfo.group_id, tmp_dot_graph_log);
  fail_op = nullptr;
  show_group(&sub_group._lgInfo);
  std::vector<std::pair<Operation *, int>> vec_op_hwsecs;
  auto max_shape_secs = get_group_max_secs(sub_group._lgInfo, vec_op_hwsecs);
  std::sort(vec_op_hwsecs.begin(), vec_op_hwsecs.end(),pair_op_int_Sort_by_int);
  std::string tmpStr = llvm::formatv("[{0},{1},{2},{3},{4}]", max_shape_secs.nsecs,
      max_shape_secs.csecs, max_shape_secs.dsecs, max_shape_secs.hsecs, max_shape_secs.wsecs).str();
  LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "max_shape_secs:" <<tmpStr<<'\n';});
  tmp_dot_graph_log->add_node_label("global_info", "max_shape_secs:" + tmpStr);
  auto& shape_secs = sub_group.shape_secs;
  std::vector<std::pair<Value, int64_t>> value_size;
  if (!init_group_data_secs2(sub_group._lgInfo, shape_secs, value_size, tmp_dot_graph_log)) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "init_group_data_secs2 fail\n";});
    is_fail_op_in_grp = false;
    fail_op = vec_op_hwsecs[0].first;
    tmp_dot_graph_log->add_node_label(module::getName(fail_op).str(),
        "init_group_data_secs2 fail");
    return false;
  }
  tmpStr = llvm::formatv("[{0},{1},{2},{3},{4}]", shape_secs.nsecs,
      shape_secs.csecs, shape_secs.dsecs, shape_secs.hsecs, shape_secs.wsecs).str();
  LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "init shape_secs:" <<tmpStr <<'\n';});
  tmp_dot_graph_log->add_node_label("global_info", "init shape_secs:" + tmpStr);

  int core_num = 1;
  if (dyn_cast<MultiCoreInterface>(BM168x::instance())) {
   core_num = module::getCoreNum();
  }
  if (core_num > 1) {
    int64_t secs = shape_secs.nsecs * shape_secs.csecs * shape_secs.dsecs * shape_secs.hsecs * shape_secs.wsecs;
    int new_secs = (secs + core_num - 1)/core_num*core_num;
    int sz = new_secs - secs;
    if (sz > 0) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() <<"algin secs:"<<secs<<" to "<<new_secs<<"\n";});
      shape_secs_t suitable_shape_secs = shape_secs;
      for (int m = 0; m < sz; m++) {
        if (!update_shape_secs_for_ilp_group(shape_secs, max_shape_secs)) {
          break;
        }
        LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "update shape shape_secs, n:" << shape_secs.nsecs
        << " c:" << shape_secs.csecs << " d:" << shape_secs.dsecs
        << " h:" << shape_secs.hsecs << " w:" << shape_secs.wsecs <<'\n';});
        int tmp_secs = shape_secs.nsecs *shape_secs.csecs *shape_secs.dsecs * shape_secs.hsecs * shape_secs.wsecs;
        if (tmp_secs > new_secs) {
          break;
        }
        suitable_shape_secs = shape_secs;
      }
      shape_secs = suitable_shape_secs;
      tmpStr = llvm::formatv("[{0},{1},{2},{3},{4}]", shape_secs.nsecs,
          shape_secs.csecs, shape_secs.dsecs, shape_secs.hsecs, shape_secs.wsecs).str();
      tmp_dot_graph_log->add_node_label("global_info", "aligned shape_secs:" + tmpStr);
    }
  }

  std::sort(value_size.begin(), value_size.end(), Sort_by_int);
  int slice_try_count = 0, max_slice_cut_count = 3;
  auto& tensor_infos = sub_group.tensor_infos;
  bool l2m_switch = module::isDebugCmdEnable("enable_l2m")?true:false;
  tmp_dot_graph_log->add_node_label("global_info", "enable_l2m");

  while (true) {
    if (++slice_try_count > max_slice_cut_count) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() <<"layer group fail\n";});
      return false; //设为global layer
    }
    int64_t secs = shape_secs.nsecs * shape_secs.csecs * shape_secs.dsecs * shape_secs.hsecs * shape_secs.wsecs;
    bool l2m_en = l2m_switch && secs > 1 && core_num > 1;
    LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "shape_secs, n:" << shape_secs.nsecs
                  << " c:" << shape_secs.csecs << " d:" << shape_secs.dsecs
                  << " h:" << shape_secs.hsecs << " w:" << shape_secs.wsecs
                  << " slice_try_count:" << slice_try_count<< " l2m_en:" << l2m_en << "\n";});

    int max_group_cycle = 0;
    sub_group.timeStepPtrs.clear();
    do {
      // tmp_dot_graph_log->export_dot("stripe_mine_idx_slice2_before");
      ret = stripe_mine_idx_slice2(sub_group._lgInfo, shape_secs, tensor_infos, fail_op);
      if(!ret){
        tmp_dot_graph_log->add_node_label(module::getName(fail_op).str(),
            "stripe_mine_idx_slice2 fail");
        LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "stripe_mine_idx_slice2 fail at "<< module::getName(fail_op).str()<<"\n";});
        if (isa<tpu::UpsampleOp>(fail_op) && shape_secs.hsecs > 1) {
          // is_fail_op_in_grp = false;
          fail_process_mode = 2;
          return false;
        }
        if (sub_group._cur_strategy != STRATEGY_SLICE_CUT_FIRST) {
          return false;
        } else {
          break;
        }
      }
      update_tensor_infos(sub_group._lgInfo, tensor_infos);
      sub_group._lgInfo.update_bank_info();

      int vec_ncdhw_idx = 0;
      std::vector<std::vector<int64_t>> vec_ncdhw;
      auto sec_per_cores = get_sec_per_cores(shape_secs, vec_ncdhw, core_num, tensor_infos);
      for (int core_id = 0; core_id < core_num; core_id++) {
        if (sec_per_cores[core_id] == 0) {
          break;
        }

        if (is_same_pipeline(core_id, sub_group.timeStepPtrs, vec_ncdhw_idx, tensor_infos, sub_group._lgInfo, vec_ncdhw, sec_per_cores)) {
          continue;
        }

        int slice_idx = 0;
        std::vector<op_var_pos_info> op_var_bound = createOverlapStrategy(sub_group._lgInfo, sec_per_cores[core_id]);
        std::map<std::string, std::string> node_labels;
        auto ilp_timeStep = std::make_shared<ILPTimeStep>(sub_group._lgInfo, tmp_dot_graph_log, sec_per_cores[core_id]);
        while(sec_per_cores[core_id]-- > 0) {
          std::vector<int64_t> ncdhw = vec_ncdhw[vec_ncdhw_idx++];
          ilp_timeStep->addSliceNcdhwSteps(core_id, ncdhw);
          LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "slice process, n:" << ncdhw[0]
                        << " c:" << ncdhw[1] << " d:" << ncdhw[2]
                        << " h:" << ncdhw[3] << " w:" << ncdhw[4]<< " ncdhw_idx:" << vec_ncdhw_idx - 1 << "\n";});
          backward_gen_ilp_var2(
              sub_group._lgInfo, shape_secs, tensor_infos, cycle_calculator_,
              *ilp_timeStep, ncdhw, slice_idx, op_var_bound,
              fail_op, node_labels, l2m_en, sec_per_cores[core_id] == 0, 4);
          slice_idx++;
        }
        if (core_id == 0) {
          for (auto itr2: node_labels) {
            tmp_dot_graph_log->add_node_label(itr2.first, itr2.second);
          }
        }
        if(!ret){
          if (sub_group._cur_strategy != STRATEGY_SLICE_CUT_FIRST) {
            return false; //里面会校验内存是否满足，需要切割
          } else {
            break;
          }
        }

        ilp_timeStep->merge_small_cycle_op(tensor_infos, tmp_dot_graph_log);
        ilp_timeStep->prepare(tensor_infos);
        // tmp_dot_graph_log->export_dot("merge_small_cycle_op_after", true);
        ret = ilp_timeStep->run(fail_op);
        if (!ret) {
          LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "ilp_timeStep->run fail\n";});
          if (fail_op) {
            tmp_dot_graph_log->add_node_label(module::getName(fail_op).str(),
              "ilp_timeStep run fail, for core_id:" + std::to_string(core_id));
            if (sub_group._cur_strategy != STRATEGY_SLICE_CUT_FIRST) {
              is_fail_op_in_grp = false;
              return false;
            }
          } else {
            tmp_dot_graph_log->add_node_label("global_info",
                "ilp_timeStep run fail, for core_id:" + std::to_string(core_id));
          }
          // fail_process_mode = 1; //求解失败则二分搜索能成功group的分割点，大组分小组
          break;
        }

        mem_alloc_status alloc_status;
        ret = ilp_timeStep->mem_alloc(alloc_status, value_size, tensor_infos, fail_op);
        if (!ret) {
          LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "ilp_timeStep->mem_alloc fail\n";});
          if (fail_op) {
            tmp_dot_graph_log->add_node_label(module::getName(fail_op).str(),
                "ilp_timeStep->mem_alloc fail, for core_id:" + std::to_string(core_id));
            if (sub_group._cur_strategy != STRATEGY_SLICE_CUT_FIRST) {
              is_fail_op_in_grp = false;
              return false;
            }
          } else {
            tmp_dot_graph_log->add_node_label("global_info",
            "ilp_timeStep->mem_alloc fail, for core_id:" + std::to_string(core_id));
          }
          break; //分配内存失败则可以考虑切成更小片
        }

        int group_cycle, group_cycle_diff;
        std::vector<std::pair<int, std::vector<Operation*>>> ts_cycle_diff;
        ilp_timeStep->get_group_cycle_info(group_cycle, group_cycle_diff, ts_cycle_diff);
        if (group_cycle > max_group_cycle) {
          max_group_cycle = group_cycle;
        }
        LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "core" << core_id<< " group_cycle" << group_cycle << ", mem_alloc success\n";});
        tmp_dot_graph_log->add_node_label("global_info", "core" + std::to_string(core_id)
            + ", group_cycle:" + std::to_string(group_cycle) + ", mem_alloc success");
        sub_group.timeStepPtrs.push_back(ilp_timeStep);
      }
    } while(false);

    if(ret){
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "ilp_timeStep success\n";});
      tmp_dot_graph_log->add_node_label("global_info", "ilp_timeStep success");
      if (fail_process_mode == 1) {
        return true;
      }
      sub_group.group_cycle = max_group_cycle;
      if (l2m_en) {
        l2m_process(sub_group, value_size);
      }
      break;
    } else {
      if (!update_shape_secs_for_ilp_group(shape_secs, max_shape_secs)) {
        return false;
      }
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() <<"update_shape_secs\n";});
      tmp_dot_graph_log->add_node_label("global_info", "update_shape_secs");
    }
  }

  return true;
}

static std::vector<std::vector<Operation *>>
ConvertDisconnectedBlocksToGroups(std::vector<Operation *> ops, std::vector<Operation*>& single_ops){
  std::vector<std::vector<Operation *>> new_grps;
  if (ops.size() < 2) {
    single_ops.insert(single_ops.end(), ops.begin(), ops.end());
    return new_grps;
  }
  LgInfo sub_group;
  sub_group.group_ops.assign(ops.begin(), ops.end());
  sub_group.update_group_io(LgPass::OPTIONS.opt);

  int in_idx = 0;
  std::map<Operation *, int> op_block_id;
  for (auto in : sub_group.group_ins) {
    for (auto user : in.getUsers()) {
      find_op_in_same_block(user, sub_group.group_ops, op_block_id, in_idx);
    }
    in_idx++;
  }

  for (int j = 0; j < in_idx; j++) {
    std::vector<Operation *> block_ops;
    for (auto itr = op_block_id.begin(); itr != op_block_id.end(); ++itr) {
      if (j == itr->second) {
        block_ops.push_back(itr->first);
      }
    }
    if (block_ops.size() > 1) {
      new_grps.push_back(block_ops);
      std::string tmpStr = "";
      for (auto op: block_ops) {
        tmpStr = tmpStr + " + " + module::getName(op).str();
      }
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "add new grp:" << tmpStr << "\n";});
    } else {
      single_ops.insert(single_ops.end(), block_ops.begin(), block_ops.end());
    }
  }
  return new_grps;
}

void GroupMethod::init_ilp_base_groups(LgPassIR* pass_ir){
  if (pass_ir->branch_parallel) {
    // get_base_branch_groups(base_groups, pass_ir->subnet_ops,
    //                        pass_ir->subnet_return_opds);
  } else {
    get_base_dfs_topo_groups(pass_ir->tmp_base_groups);
  }
}
int ilp_LgInfo::group_count = 0;
void ilp_LgInfo::save_result(LgPassIR *pass_ir) {
  pass_ir->ILP_time_steps.push_back(timeStepPtrs);
  pass_ir->shape_secs.push_back(shape_secs);
  pass_ir->lg_tensor_infos_.push_back(tensor_infos);
  pass_ir->lg_infos.push_back(_lgInfo);
  pass_ir->group_cycles.push_back(group_cycle);
  pass_ir->map_l2m_loads.push_back(map_l2m_load);
  pass_ir->lg_l2mem_alloc_ptr.push_back(l2mem_alloc);
}

std::shared_ptr<ilp_LgInfo> ilp_LgInfo::high_solver(LgPassIR *pass_ir, std::shared_ptr<CycleCalculator> cycle_calculator_) {
  auto ops_ori = _lgInfo.group_ops;
  _cur_strategy = STRATEGY_GROUP_CUT_FIRST;
  LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::errs()<<"ilp_debug: STRATEGY_GROUP_CUT_FIRST test\n";});
  base_solver(pass_ir, cycle_calculator_);
  if (module::isDebugCmdEnable("enable_high_solver")) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::errs()<<"ilp_debug: STRATEGY_SLICE_CUT_FIRST test\n";});
    auto ilp_cloned = CreateIlpLgInfo(ops_ori, STRATEGY_SLICE_CUT_FIRST);
    ilp_cloned->base_solver(pass_ir, cycle_calculator_);
    if (group_cycle > ilp_cloned->group_cycle) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::errs()<<"ilp_debug:strategy STRATEGY_SLICE_CUT_FIRST better, "
        <<group_cycle<<" vs "<<ilp_cloned->group_cycle<<"\n";});
      return ilp_cloned;
    } else {
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::errs()<<"ilp_debug:strategy STRATEGY_GROUP_CUT_FIRST better, "
        <<group_cycle<<" vs "<<ilp_cloned->group_cycle<<"\n";});
    }
  }
  return nullptr;
}

void ilp_LgInfo::base_solver(LgPassIR *pass_ir, std::shared_ptr<CycleCalculator> cycle_calculator_) {
  auto& ops = _lgInfo.group_ops;
  auto tmp_dot_graph_log = pass_ir->dot_graph_log_subnet->clone();
  for (auto [index, op] : llvm::enumerate(ops)) {
    tmp_dot_graph_log->add_node_label(module::getName(op).str(),
      "grp_ts" + std::to_string(index + 1) +"*");
  }
  ilp_func_trace tmp_trace(__func__);
  int fail_process_mode = 0;
  Operation* fail_op = nullptr;
  _lgInfo.update_group_io(LgPass::OPTIONS.opt);
  std::map<Operation*, bool> break_op_reside;
  std::map<Operation*, bool>* break_op_reside_ptr = nullptr;
  std::vector<Operation *> break_ops;

  bool is_fail_op_in_grp = true;
  auto ret = ilp_for_single_group(pass_ir, *this, fail_process_mode,
                                  fail_op, is_fail_op_in_grp, cycle_calculator_);
  if (!ret) {
    if (_cur_strategy == STRATEGY_SEARCH_CONV_CUT) {
      return; //搜索模式下不再嵌套group
    }
    if (fail_op && fail_process_mode == 0) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::errs()<<"ilp_debug: ilp_for_single_group fail_op:"
        <<show_op_info(fail_op)<<"\n";});
      break_op_reside[fail_op] = is_fail_op_in_grp;
      break_op_reside_ptr = &break_op_reside;
      break_ops.push_back(fail_op);
      if (!is_fail_op_in_grp) {
        global_layers.push_back(fail_op);
      }
    }  else if (fail_process_mode == 2) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::errs()<<"ilp_debug: fail_process_mode 2\n";});
      if (isOpTypeInGroup<tpu::UpsampleOp>(ops, break_ops)) {
        for (auto op: break_ops) {
          LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::errs()<<"ilp_debug: break_op:"<<show_op_info(op)<<"\n";});
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
    for (auto [i, grp] : llvm::enumerate(seg_grp_ops_by_global_op(ops, break_ops, break_op_reside_ptr))) {
      if (grp.size() > 1) {
        ilp_func_trace tmp_trace(llvm::formatv("ilp_debug: process_sub_group, i:{0}", i).str());
        auto tmpLgInfo= CreateIlpLgInfo(sortOpsByOtherOpsOrder(_lgInfo.group_ops, grp));
        tmpLgInfo->base_solver(pass_ir, cycle_calculator_);
        group_cycle += tmpLgInfo->group_cycle;
        sub_ilp_LgInfos.push_back(tmpLgInfo);
      } else {
        LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::errs()<<"ilp_debug: add global_layer:"<<show_op_info(grp[0])<<"\n";});
        global_layers.push_back(grp[0]);
      }
    }
  }
  for (auto global_layer: global_layers) {
    group_cycle += cycle_calculator_->getGlobalLayerCycle(global_layer);
  }
}

bool ilp_LgInfo::binary_search_group(bool move_right, std::shared_ptr<dot_graph> dot_graph_log) {
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
      for (auto op: group_ops_all) {
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
  while(true) {
    auto nodes = GetParallelNodes(group_ops_all[middle_ptr]);
    if (!nodes.size()) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() <<"ParallelNodes is None, middle_ptr ok\n";});
      break;
    }
    if (++middle_ptr >= right_ptr - 1) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() <<"inc middle_ptr\n";});
      break;
    }
  }

  if (move_right && pre_middle_ptr != -1) {
    last_success_middle_ptr = pre_middle_ptr;
  }
  auto cut_op = group_ops_all[middle_ptr];
  LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() <<"ilp_debug: binary_search_group, middle_ptr:"<<middle_ptr
               <<", op(excluded):"<<show_op_info(cut_op)<< "\n";});
  auto name = module::getName(cut_op).str();
  dot_graph_log->add_node_label(name + "_ori", std::string("binary_search_cut_op"));
  int i = 0;
  _lgInfo.group_ops.clear();
  for (auto op: group_ops_all) {
    if (i < middle_ptr) {
      _lgInfo.group_ops.push_back(op);
    }
    i++;
  }
  _lgInfo.update_group_io(LgPass::OPTIONS.opt);
  set_group_type(_lgInfo);
  pre_middle_ptr = middle_ptr;
  return true;
}

std::vector<Operation*> ilp_LgInfo::GetParallelNodes(Operation* op) {
  if (!map_parallel_node.size()) {
    GetAllParallelNodes(_lgInfo.group_ops, map_parallel_node, &_lgInfo.group_ops);
  }
  return map_parallel_node[op];
}

static void collectAllSubLgInfoResult(std::shared_ptr<ilp_LgInfo> lgInfo, LgPassIR *pass_ir) {
  if (lgInfo->group_success) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs()<<"add group to LgPassIR:"<<lgInfo->_lgInfo.group_id<<"\n";});
    lgInfo->save_result(pass_ir);
  }

  for (auto it: lgInfo->sub_ilp_LgInfos) {
    if (it->group_success) {
      LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs()<<"add group to LgPassIR:"<<it->_lgInfo.group_id<<"\n";});
      llvm::outs()<<"add group to LgPassIR:"<<it->_lgInfo.group_id<<"\n";
      it->save_result(pass_ir);
    } else {
      collectAllSubLgInfoResult(it, pass_ir);
    }
  }
}

static void collectAllSubLgInfo(std::shared_ptr<ilp_LgInfo> lgInfo,
            std::vector<std::shared_ptr<ilp_LgInfo>>& base_groups) {
  if (lgInfo->group_success) {
    base_groups.push_back(lgInfo);
  }

  for (auto sub_lgInfo: lgInfo->sub_ilp_LgInfos) {
    if (sub_lgInfo->group_success) {
      base_groups.push_back(sub_lgInfo);
    } else {
      collectAllSubLgInfo(sub_lgInfo, base_groups);
    }
  }
}


static std::shared_ptr<std::vector<std::shared_ptr<ilp_LgInfo>>>
expandAllNestedLgInfo(std::vector<std::shared_ptr<ilp_LgInfo>>& base_groups) {
  auto new_base_groups = std::make_shared<std::vector<std::shared_ptr<ilp_LgInfo>>>();
  for (int64_t i = 0, grp_num = base_groups.size(); i < grp_num; i++) {
    collectAllSubLgInfo(base_groups[i], *new_base_groups);
  }
  return std::move(new_base_groups);
}



void GroupMethod::ilp_layer_group(LgPassIR *pass_ir) {
  LAYER_GROUP_LOG_DEBUG_BLOCK({llvm::outs() << "\n"
               << "=======================================================\n"
               << "*********** ilp_layer_group **********\n"
               << "=======================================================\n";});
  std::vector<Operation*> subnet_ops;
  for (auto it: pass_ir->subnet_ops) {
    subnet_ops.push_back(it);
  }
  pass_ir->dot_graph_log_subnet = createSubnetGraph(subnet_ops);
  //------------------------part0: pre processing----------------------------------------------------
  init_ilp_base_groups(pass_ir);
  pass_ir->dot_graph_log_subnet->add_node_label("global_info",
    "init group_num:" + std::to_string(pass_ir->tmp_base_groups.size()));
  // pass_ir->dot_graph_log_subnet->export_dot("svg_initial_" + module::getName(module::getModuleOp()).str(), true);

  //------------------------part1: processing----------------------------------------------------
  std::vector<std::shared_ptr<ilp_LgInfo>> base_groups2;
  for (int64_t i = 0, grp_num = pass_ir->tmp_base_groups.size(); i < grp_num; i++) {
    ilp_func_trace tmp_trace(llvm::formatv("high_solver, i:{0}", i).str());
    auto best_lgInfo = pass_ir->tmp_base_groups[i]->high_solver(pass_ir, cycle_calculator_);
    base_groups2.push_back(best_lgInfo?best_lgInfo:pass_ir->tmp_base_groups[i]);
  }

  auto base_groups3 = expandAllNestedLgInfo(base_groups2);
  try_cut_some_group(pass_ir, *base_groups3); //优先大的group，放到后面去处理

  auto base_groups4 = expandAllNestedLgInfo(*base_groups3);
  for (int64_t i = 0, grp_num = base_groups4->size(); i < grp_num; i++) {
    collectAllSubLgInfoResult((*base_groups4)[i], pass_ir);
  }

  pass_ir->dot_graph_log_subnet->add_node_label("global_info",
    "final group_num:" + std::to_string(pass_ir->lg_infos.size()));
  for (auto [grp_idx, lg_info] : llvm::enumerate(pass_ir->lg_infos)) {
    for (auto [op_idx, op] : llvm::enumerate(lg_info.group_ops)) {
      pass_ir->dot_graph_log_subnet->add_node_label(module::getName(op).str(),
            "grp_" + std::to_string(grp_idx) + "*_id_" + std::to_string(lg_info.group_id) + "*_" + std::to_string(op_idx) +"*");
    }
  }
  pass_ir->dot_graph_log_subnet->export_dot("svg_" + module::getName(module::getModuleOp()).str());
}

void GroupMethod::process(LgPassIR *pass_ir) {
  std::vector<LgInfo> &lg_infos = pass_ir->lg_infos;
  llvm::SetVector<Operation *> &subnet_ops = pass_ir->subnet_ops;
  auto start = std::chrono::high_resolution_clock::now();
  runmode_ = getRunMode(subnet_ops[0]);
  auto func_name = pass_ir->func.getName();

  switch (LgPass::OPTIONS.opt) {
  case 1:
    simple_layer_group(lg_infos, subnet_ops);
    dump_cut_results(func_name);
    break;
  case 2:
    dynamic_programming_layer_group_with_cluster(lg_infos, subnet_ops);
    dump_cut_results(func_name);
    break;
  case 3:
    ilp_layer_group(pass_ir);
    dump_cut_results(func_name);
    break;
  case 4: {
    if(is_cut_results_exists(func_name)){
      load_cut_results(func_name);
      show_cut_results();
      std::vector<std::vector<Operation *>> base_groups;
      get_base_groups(base_groups, subnet_ops);
      get_final_groups(lg_infos, base_groups);
    }else{
      llvm_unreachable("cut_results.txt not exist s, ues opt=1/2/3 to generate");
    }
  } break;
  default:
    simple_layer_group(lg_infos, subnet_ops);
    break;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
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
      get_layer_group(lg_info, base_group, start_idx, end_idx);
      if (lg_info.group_ops.size() > 1 ||
          false == LgPass::OPTIONS.group_by_cores) {
        lg_infos.push_back(lg_info);
      }
      DEBUG_WITH_TYPE("lg_results", {
        if(runmode_ == RunMode::TPU_STATIC){
          int64_t cost = 0;
          is_layer_group_valid(lg_info, true, &cost);
          llvm::dbgs() << "; action = lg_results"
          << "; start_idx = " << start_idx
                        << "; end_idx = " << end_idx
                        << "; group_cost = " << cost
                        << "; final_group_idx = " << i
                        << "\n";
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
  }});

}


bool GroupMethod::is_cut_results_exists(StringRef func_name){
    return std::filesystem::exists("cut_results_" + func_name.str() + ".mlircache");

}
void GroupMethod::dump_cut_results(StringRef func_name) {
  if(!LgPass::OPTIONS.lgcache){
     return;
  }
  std::ofstream out("cut_results_" + func_name.str() + ".mlircache");
  if (!out.is_open()) {
    std::cerr << "Failed to open file for writing.\n";
    return;
  }
  out << opt_ << "\n";
  for (const auto &row : cut_results_) {
    for (const auto &item : row) {
      out << item << " ";
    }
    out << "\n"; // 每个内部vector结束后换行
  }

  out.close();
}

void GroupMethod::load_cut_results(StringRef func_name) {
  std::ifstream in("cut_results_" + func_name.str() + ".mlircache");
  if (!in.is_open()) {
    std::cerr << "Failed to open file for reading.\n";
    return;
  }

  cut_results_.clear(); // 清空现有数据
  std::string line;

  in >> opt4_ori_opt_;
  in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  // 忽略到行尾，准备读取下一行

  while (std::getline(in, line)) {

    std::istringstream iss(line);
    std::vector<int64_t> row;
    int64_t value;

    while (iss >> value) {
      row.push_back(value);
    }

    cut_results_.push_back(row);
  }

  in.close();
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

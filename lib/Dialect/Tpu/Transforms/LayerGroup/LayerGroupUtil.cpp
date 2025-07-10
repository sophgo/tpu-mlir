//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgPass.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "llvm/Support/FormatVariadic.h"

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/IlpTimeStep.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/TimeStepMethod.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "LayerGroupUtil"

using namespace tpu_mlir::backend;

namespace tpu_mlir {
namespace tpu {

std::string show_op_info(Operation *op) {
  if (!op) {
    return "";
  }
  auto name = module::getName(op).str();
  return llvm::formatv("  op: {0}, type: {1}", name, op->getName()).str();
}

std::string shape_str(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w) {
  return llvm::formatv("[{0},{1},{2},{3},{4}]", n, c, d, h, w);
}
std::string shape_str(std::vector<int64_t> ncdhw) {
  for (int i = 0; i < 5 - ncdhw.size(); i++) {
    ncdhw.push_back(-1);
  }
  return shape_str(ncdhw[0], ncdhw[1], ncdhw[2], ncdhw[3], ncdhw[4]);
}
int64_t align(int64_t input, int64_t align_size) {
  // if (input % align_size != 0) {
  //   fprintf(stderr, "warning, input:%ld is not align %ld\n", input,
  //   align_size);
  // }
  return (int64_t)((input + align_size - 1) / align_size * align_size);
}

int64_t align_64(int64_t input) { return (int64_t)((input + 63) / 64 * 64); }

bool grp_is_valid(std::vector<Operation *> &group_ops) {
  int valid_op_count = 0;
  for (auto op : group_ops) {
    if (op && !isa<tpu::ReshapeOp>(op)) {
      valid_op_count++;
    }
  }
  return valid_op_count > 1;
}
void show_group(const LgInfo *sub_group) {
  if (!module::isDebugCmdEnable("detail_info_show")) {
    return;
  }

  if (sub_group->group_ops.size() == 0)
    return;

  llvm::errs() << "group_ops, size:" << sub_group->group_ops.size() << "\n";
  for (auto op : sub_group->group_ops) {
    if (op) {
      llvm::errs() << show_op_info(op) << "\n";
    }
  }
  for (auto out : sub_group->group_outs) {
    llvm::errs() << "    out:" << module::getName(out).str() << "\n";
  }
  for (auto in : sub_group->group_ins) {
    llvm::errs() << "    in:" << module::getName(in).str() << "\n";
  }
}

void PrintOps(std::string ops_name, const std::vector<Operation *> &ops) {
  std::string tmpStr = "";
  for (auto op : ops) {
    tmpStr = tmpStr + " + " + show_op_info(op);
  }
  if (ops.size()) {
    tmpStr = tmpStr.substr(3);
  }
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::errs() << "ops_name:" << ops_name << ",ops:" << tmpStr << "\n";
  });
}

bool isLgSupport(Operation *op) {
  bool res = false;
  if (isa<top::WeightOp, top::InputOp>(op)) {
    res = true;
  }
  if (auto lg_op = dyn_cast<tpu_mlir::LocalGenInterface>(op)) {
    res = mlir::succeeded(lg_op.LocalGenSupport());
  }
  return res;
}

std::vector<Operation *>
sortOpsByOtherOpsOrder(const std::vector<Operation *> &exp_ops,
                       const std::vector<Operation *> &ops) {
  std::vector<Operation *> tmp_ops;
  for (auto op : exp_ops) {
    if (std::find(ops.begin(), ops.end(), op) != ops.end()) {
      tmp_ops.push_back(op);
    }
  }
  return std::move(tmp_ops);
}

bool isCheckOpInOtherOps(std::vector<Operation *> &check_ops,
                         std::vector<Operation *> &other_ops) {
  for (auto op : check_ops) {
    if (std::find(other_ops.begin(), other_ops.end(), op) != other_ops.end()) {
      llvm::errs() << "check_op is already in other_ops\n";
      return false;
    }
  }
  return true;
}

inline bool opIsInGrpOps(Operation *op, std::vector<Operation *> grp_ops) {
  if (std::find(grp_ops.begin(), grp_ops.end(), op) != grp_ops.end()) {
    return true;
  }
  return false;
}

void find_op_tree_by_root2(Operation *op, std::vector<Operation *> &op_tree,
                           const std::vector<Operation *> &ops,
                           const std::vector<Operation *> &exclude_ops,
                           const std::vector<Operation *> &break_ops,
                           int cur_depth, int max_depth) {
  op_tree.push_back(op);
  if (max_depth > 0 && cur_depth > max_depth) {
    return;
  }
  int tmp_depth = cur_depth + 1;
  for (auto user : op->getUsers()) {
    if (!isa<ReturnOp>(user) &&
        std::find(ops.begin(), ops.end(), user) != ops.end() &&
        std::find(exclude_ops.begin(), exclude_ops.end(), user) ==
            exclude_ops.end() &&
        std::find(op_tree.begin(), op_tree.end(), user) == op_tree.end() &&
        std::find(break_ops.begin(), break_ops.end(), user) ==
            break_ops.end()) {
      find_op_tree_by_root2(user, op_tree, ops, exclude_ops, break_ops,
                            tmp_depth, max_depth);
    }
  }
}

static std::vector<std::vector<Operation *>>
process_in_value_have_mulit_user(const std::vector<Operation *> &ops,
                                 const LgOptions &options) {
  LgInfo lgInfo;
  lgInfo.group_ops.assign(ops.begin(), ops.end());
  lgInfo.update_group_io(options.opt);
  std::vector<Value> multi_user_values;
  for (auto in : lgInfo.group_ins) {
    if (valueHasMultiGroupUser(in, ops)) {
      multi_user_values.push_back(in);
    }
  }

  std::vector<std::vector<Operation *>> new_grps;
  if (multi_user_values.size() > 0) {
    std::vector<Operation *> accessed_ops, left_ops, break_ops;
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
      auto op = *it;
      for (auto opd : op->getOperands()) {
        if (std::find(multi_user_values.begin(), multi_user_values.end(),
                      opd) != multi_user_values.end()) {
          std::vector<Operation *> op_tree;
          find_op_tree_by_root2(op, op_tree, ops, accessed_ops, break_ops);
          if (op_tree.size() > 1) {
            new_grps.push_back(op_tree);
          } else {
            // if (op_tree.size() == 1) {
            //   llvm::errs() << "op_tree only have:" <<
            //   module::getName(op_tree[0]).str() << "\n";
            // }
          }
          accessed_ops.insert(accessed_ops.end(), op_tree.begin(),
                              op_tree.end());
        }
      }
    }
    for (auto op : ops) {
      if (!isa<ReturnOp>(op) &&
          std::find(accessed_ops.begin(), accessed_ops.end(), op) ==
              accessed_ops.end()) {
        left_ops.push_back(op);
      }
    }
    if (left_ops.size() > 1) {
      new_grps.push_back(left_ops);
    } else {
      if (left_ops.size() == 1) {
        // llvm::errs() << "left_ops only have:" <<
        // module::getName(left_ops[0]).str() << "\n";
      }
    }
  } else {
    new_grps.push_back(ops);
  }
  return std::move(new_grps);
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

static std::vector<std::vector<Operation *>>
ConvertDisconnectedBlocksToGroups(std::vector<Operation *> ops,
                                  std::vector<Operation *> &single_ops,
                                  const LgOptions &options) {
  std::vector<std::vector<Operation *>> new_grps;
  if (ops.size() < 2) {
    if (ops.size() == 1) {
      // llvm::errs() << "add to single_ops:" << module::getName(ops[0]).str()
      // << "\n";
    }
    single_ops.insert(single_ops.end(), ops.begin(), ops.end());
    return new_grps;
  }
  LgInfo sub_group;
  sub_group.group_ops.assign(ops.begin(), ops.end());
  sub_group.update_group_io(options.opt);

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
      for (auto op : block_ops) {
        tmpStr = tmpStr + " + " + module::getName(op).str();
      }
      LAYER_GROUP_LOG_DEBUG_BLOCK(
          { llvm::errs() << "add new grp:" << tmpStr << "\n"; });
    } else {
      if (block_ops.size() == 1) {
        // llvm::errs() << "add to single_ops2:" <<
        // module::getName(block_ops[0]).str() << "\n";
      }
      single_ops.insert(single_ops.end(), block_ops.begin(), block_ops.end());
    }
  }
  return new_grps;
}

void findReshapeAtEdge(std::vector<Operation *> &ops,
                       std::vector<Operation *> &del_ops) {
  for (auto op : ops) {
    if (isa<tpu::ReshapeOp>(op)) {
      bool in_grp = false;
      for (auto user : op->getUsers()) {
        if (std::find(ops.begin(), ops.end(), user) != ops.end()) {
          in_grp = true;
          break;
        }
      }
      if (!in_grp) {
        del_ops.push_back(op);
      }

      in_grp = false;
      for (auto v : op->getOperands()) {
        if (std::find(ops.begin(), ops.end(), v.getDefiningOp()) != ops.end()) {
          in_grp = true;
          break;
        }
      }
      if (!in_grp) {
        del_ops.push_back(op);
      }
    }
  }
}

std::vector<std::vector<Operation *>>
seg_grp_ops_by_global_op(const std::vector<Operation *> &grp_ops,
                         const std::vector<Operation *> &break_ops,
                         std::vector<Operation *> &excluded_ops,
                         const LgOptions &options,
                         std::map<Operation *, bool> *break_op_reside) {
  std::vector<std::vector<Operation *>> new_grps, new_grps2;
  std::vector<Operation *> left_ops;
  for (auto it = grp_ops.rbegin(); it != grp_ops.rend(); ++it) {
    auto op = *it;
    if (!op)
      continue;
    if (std::find(excluded_ops.begin(), excluded_ops.end(), op) ==
        excluded_ops.end()) {
      if (std::find(break_ops.begin(), break_ops.end(), op) !=
          break_ops.end()) {
        if (break_op_reside &&
            (*break_op_reside).find(op) != (*break_op_reside).end() &&
            (*break_op_reside)[op]) {
          std::vector<Operation *> op_tree;
          find_op_tree_by_root2(op, op_tree, grp_ops, excluded_ops, break_ops);
          new_grps.push_back(op_tree);
          excluded_ops.insert(excluded_ops.end(), op_tree.begin(),
                              op_tree.end());
        } else {
          for (auto user : op->getUsers()) {
            if (!isa<ReturnOp>(user) &&
                std::find(grp_ops.begin(), grp_ops.end(), user) !=
                    grp_ops.end() &&
                std::find(break_ops.begin(), break_ops.end(), user) ==
                    break_ops.end()) {
              std::vector<Operation *> op_tree;
              find_op_tree_by_root2(user, op_tree, grp_ops, excluded_ops,
                                    break_ops);
              // llvm::errs()<<"find new grp, user:"<<show_op_info(user)<<",
              // root op:"<<show_op_info(op)<<"\n"; for (auto op2: op_tree) {
              //   if (isa<tpu::SoftmaxOp, tpu::MatMulOp>(op2)) {
              //     llvm::errs()<<"find the:"<<show_op_info(op2)<<"\n";
              //   }
              // }
              new_grps.push_back(op_tree);
              excluded_ops.insert(excluded_ops.end(), op_tree.begin(),
                                  op_tree.end());
            }
          }
        }
      }
    }
  }
  for (auto op : grp_ops) {
    if (op && !isa<ReturnOp>(op) &&
        std::find(excluded_ops.begin(), excluded_ops.end(), op) ==
            excluded_ops.end() &&
        std::find(break_ops.begin(), break_ops.end(), op) == break_ops.end()) {
      // if (isa<tpu::SoftmaxOp, tpu::MatMulOp>(op)) {
      //   llvm::errs()<<"find the2:"<<show_op_info(op)<<"\n";
      // }
      left_ops.push_back(op);
    }
  }
  std::vector<Operation *> single_ops;
  auto tmpGrps =
      ConvertDisconnectedBlocksToGroups(left_ops, single_ops, options);
  new_grps.insert(new_grps.end(), tmpGrps.begin(), tmpGrps.end());

  for (auto grp : new_grps) {
    if (grp.size() > 1) {
      for (auto grp2 : process_in_value_have_mulit_user(grp, options)) {
        if (grp2.size() > 1) {
          new_grps2.push_back(grp2);
        }
      }
    }
  }

  for (auto &ops : new_grps2) {
    auto tmp_ops = sortOpsByOtherOpsOrder(grp_ops, ops);
    ops.assign(tmp_ops.begin(), tmp_ops.end());
  }

  for (auto &ops : new_grps2) {
    std::vector<Operation *> del_ops;
    findReshapeAtEdge(ops, del_ops);
    for (auto del_op : del_ops) {
      llvm::errs() << "seg_grp_ops_by_global_op findReshapeAtEdge: "
                   << module::getName(del_op).str() << "\n";
      ops.erase(std::remove(ops.begin(), ops.end(), del_op), ops.end());
    }
  }

  return std::move(new_grps2);
}

std::vector<std::vector<Operation *>>
seg_network_by_group_ops(const std::vector<Operation *> &network_ops,
                         const std::vector<Operation *> &group_ops) {
  LgInfo lgInfo;
  lgInfo.group_ops.assign(group_ops.begin(), group_ops.end());
  lgInfo.update_group_io();
  std::vector<Operation *> down_sub_ops, tmp_down_sub_ops, up_sub_ops;
  for (auto out : lgInfo.group_outs) {
    for (auto user : out.getUsers()) {
      std::vector<Operation *> op_tree, tmp_ops;
      find_op_tree_by_root2(user, op_tree, network_ops, tmp_ops, tmp_ops);
      tmp_down_sub_ops.insert(tmp_down_sub_ops.end(), op_tree.begin(),
                              op_tree.end());
    }
  }

  std::sort(tmp_down_sub_ops.begin(), tmp_down_sub_ops.end());
  auto last = std::unique(tmp_down_sub_ops.begin(), tmp_down_sub_ops.end());
  tmp_down_sub_ops.erase(last, tmp_down_sub_ops.end());
  // removeDuplicateOp(tmp_down_sub_ops);
  for (auto op : tmp_down_sub_ops) {
    if (std::find(group_ops.begin(), group_ops.end(), op) == group_ops.end()) {
      down_sub_ops.push_back(op);
    }
  }

  for (auto op : network_ops) {
    if (std::find(group_ops.begin(), group_ops.end(), op) == group_ops.end() &&
        std::find(down_sub_ops.begin(), down_sub_ops.end(), op) ==
            down_sub_ops.end()) {
      up_sub_ops.push_back(op);
    }
  }
  std::vector<std::vector<Operation *>> new_grps;
  new_grps.push_back(up_sub_ops);
  new_grps.push_back(down_sub_ops);

  for (auto &ops : new_grps) {
    auto tmp_ops = sortOpsByOtherOpsOrder(network_ops, ops);
    ops.assign(tmp_ops.begin(), tmp_ops.end());
  }
  return std::move(new_grps);
}

void find_all_pre_ops(Operation *op, std::vector<Operation *> &glayer_pre_ops,
                      std::vector<Operation *> *grp_ops) {
  if (grp_ops && !opIsInGrpOps(op, *grp_ops)) {
    return;
  }
  glayer_pre_ops.push_back(op);
  for (auto [i, v] : llvm::enumerate(op->getOperands())) {
    auto pre_op = v.getDefiningOp();
    if (pre_op == nullptr ||
        isa<top::NoneOp, top::WeightOp, top::InputOp>(pre_op)) {
      continue;
    }
    // if (i > 0 && !isa<tpu::AddOp, tpu::ConcatOp>(pre_op)) {
    //   continue;
    // }
    if (grp_ops && !opIsInGrpOps(pre_op, *grp_ops)) {
      continue;
    }
    if (std::find(glayer_pre_ops.begin(), glayer_pre_ops.end(), pre_op) ==
        glayer_pre_ops.end()) {
      find_all_pre_ops(pre_op, glayer_pre_ops, grp_ops);
    }
  }
}

void find_all_next_ops(Operation *op, std::vector<Operation *> &glayer_next_ops,
                       std::vector<Operation *> *grp_ops) {
  if (grp_ops && !opIsInGrpOps(op, *grp_ops)) {
    return;
  }

  glayer_next_ops.push_back(op);
  for (auto user : op->getUsers()) {
    if (isa<ReturnOp>(user)) {
      continue;
    }

    if (grp_ops && !opIsInGrpOps(user, *grp_ops)) {
      continue;
    }

    if (std::find(glayer_next_ops.begin(), glayer_next_ops.end(), user) ==
        glayer_next_ops.end()) {
      find_all_next_ops(user, glayer_next_ops, grp_ops);
    }
  }
}

std::shared_ptr<ilp_LgInfo>
CreateIlpLgInfo(std::vector<Operation *> ops, const LgOptions &options,
                solver_strategy_type_t cur_strategy) {
  auto ilp_lgInfo = std::make_shared<ilp_LgInfo>();
  ilp_lgInfo->_cur_strategy = cur_strategy;
  ilp_lgInfo->_lgInfo.group_ops.assign(ops.begin(), ops.end());
  ilp_lgInfo->_lgInfo.update_group_io(options.opt);
  ilp_lgInfo->options_.dyn_compile = options.dyn_compile;
  ilp_lgInfo->options_.opt = options.opt;
  // set_group_type(ilp_lgInfo->_lgInfo);
  llvm::errs() << "add_group_id:" << ilp_lgInfo->_lgInfo.group_id << "\n";
  return ilp_lgInfo;
}

void GetAllParallelNodes(
    const std::vector<Operation *> &ops,
    std::map<Operation *, std::vector<Operation *>> &map_parallel_node,
    std::vector<Operation *> *grp_ops) {
  for (auto op : ops) {
    std::vector<Operation *> dep_ops, tmp_ops;
    find_all_pre_ops(op, dep_ops, grp_ops);
    find_all_next_ops(op, dep_ops, grp_ops);
    for (auto itr : ops) {
      if (std::find(dep_ops.begin(), dep_ops.end(), itr) == dep_ops.end()) {
        tmp_ops.push_back(itr);
      }
    }
    map_parallel_node[op] = tmp_ops;
  }
}

bool opHasMultiGroupUser(Operation *op,
                         const std::vector<Operation *> &grp_ops) {
  int count = 0;
  for (auto user : op->getUsers()) {
    if (isa<ReturnOp>(user)) {
      continue;
    }

    if (std::find(grp_ops.begin(), grp_ops.end(), user) != grp_ops.end()) {
      count++;
    }
  }
  return count > 1 ? true : false;
}

bool valueHasMultiGroupUser(Value value,
                            const std::vector<Operation *> &grp_ops) {
  int count = 0;
  for (auto user : value.getUsers()) {
    if (isa<ReturnOp>(user)) {
      continue;
    }

    if (std::find(grp_ops.begin(), grp_ops.end(), user) != grp_ops.end()) {
      count++;
    }
  }
  return count > 1 ? true : false;
}

std::string replaceChars_for_dot(std::string str) {
  if (std::isdigit(str[0])) {
    str = "s_" + str;
  }
  std::string toReplace = "./:";
  for (size_t i = 0; i < toReplace.size(); ++i) {
    std::replace(str.begin(), str.end(), toReplace[i], '_');
  }
  return str;
}

bool isPreOpHaveAComputeOp(Operation *op) {
  if (!isa<tpu::ReshapeOp, tpu::PermuteOp>(op)) {
    LLVM_DEBUG(llvm::dbgs()
                   << "meet compute op:" << module::getName(op).str() << "\n";);
    return true;
  }
  LLVM_DEBUG(llvm::dbgs() << "meet not_compute op:" << module::getName(op).str()
                          << "\n";);
  bool have_compute = false;
  for (auto v : op->getOperands()) {
    auto pre_op = v.getDefiningOp();
    if (pre_op == nullptr ||
        isa<top::NoneOp, top::WeightOp, top::InputOp>(pre_op)) {
      continue;
    }
    if (isPreOpHaveAComputeOp(pre_op)) {
      have_compute = true;
      break;
    }
  }
  return have_compute;
}

shape_secs_t
get_group_max_secs(const LgInfo &lg_info,
                   std::vector<std::pair<Operation *, int>> &vec_op_hwsecs) {
  int64_t n, c, d, h, w;
  module::getNCDHW(lg_info.group_ops[0]->getOperand(0), n, c, d, h, w,
                   lg_info.type);
  int64_t max_nsecs = n;
  if (isa<tpu::AddOp, tpu::SubOp, tpu::MulOp, tpu::DivOp, tpu::MaxOp,
          tpu::MinOp, tpu::MatMulOp>(lg_info.group_ops[0])) {
    module::getNCDHW(lg_info.group_ops[0]->getOperand(1), n, c, d, h, w,
                     lg_info.type);
    // if (isa<tpu::MatMulOp>(lg_info.group_ops[0]) && n != max_nsecs) {
    //   max_nsecs = 1;
    // } else {
    //   max_nsecs = std::max(n, max_nsecs);
    // }
    // // allow MatMul split by n
    max_nsecs = std::max(n, max_nsecs);
  }
  int64_t max_csecs = llvm::maxIntN(64);
  int64_t max_hsecs = llvm::maxIntN(64);
  int64_t max_dsecs = llvm::maxIntN(64);
  int64_t max_wsecs = llvm::maxIntN(64);
  // Need consider n_align if backend is BM1684
  int64_t n_align = 1;
  for (auto op : lg_info.group_ops) {
    if (!op) {
      continue;
    }
    auto mode = getRunMode(dyn_cast<func::FuncOp>(op->getParentOp()));
    auto lgOp = dyn_cast<LocalGenInterface>(op);
    int min_total_secs = 100000;
    for (auto v : get_output_values(op)) {
      module::getNCDHW(v, n, c, d, h, w, lg_info.type);
      auto stype = module::getStorageType(v);
      int dtype_bytes = stype.getIntOrFloatBitWidth() / 8, total_secs = 1;
      if (Arch::ALIGN_4N) {
        auto stype = module::getStorageType(v);
        n_align = 32 / stype.getIntOrFloatBitWidth();
      }
      if (succeeded(lgOp.AllowDataSplit(0, lg_info.type))) {
        auto nsecs = ceiling_func(n, n_align);
        max_nsecs = std::min(max_nsecs, nsecs);
        total_secs *= nsecs;
      } else {
        max_nsecs = 1;
      }

      if (mode != RunMode::TPU_DYNAMIC &&
          (lg_info.type == GROUP_MM || lg_info.type == GROUP_SMALL_C ||
           (lg_info.type == GROUP_NORMAL &&
            (isa<tpu::BatchNormBwdOp>(op) || isa<tpu::WhereOp>(op)))) &&
          succeeded(lgOp.AllowDataSplit(1, lg_info.type))) {
        int64_t csecs = ceiling_func(c, Arch::NPU_NUM);
        max_csecs = std::min(max_csecs, csecs);
        total_secs *= csecs;
      } else {
        max_csecs = 1;
      }

      // if (max_nsecs * max_csecs < module::getCoreNum()) {
      //   max_csecs = module::getCoreNum();
      // }

      // split d now only supports BM1684X and not int4, not dynamic
      if ((module::isBM1684XFamily() || module::isBM1690Family()) &&
          (!stype.isInteger(4)) && lg_info.type == GROUP_3D &&
          mode != RunMode::TPU_DYNAMIC &&
          succeeded(lgOp.AllowDataSplit(2, lg_info.type))) {
        max_dsecs = std::min(max_dsecs, d);
        total_secs *= d;
      } else {
        max_dsecs = 1;
      }
      if (succeeded(lgOp.AllowDataSplit(2 + (lg_info.type == GROUP_3D ? 1 : 0),
                                        lg_info.type))) {
        max_hsecs = std::min(max_hsecs, h);
        total_secs *= h;
      } else {
        // llvm::errs() << "can not split h at op:" << module::getName(op).str()
        // << "\n";
        max_hsecs = 1;
      }
      // split w now only supports BM1684X and not int4, not dynamic
      if ((module::isBM1684XFamily() || module::isBM1690Family()) &&
          (!stype.isInteger(4)) && mode != RunMode::TPU_DYNAMIC &&
          succeeded(lgOp.AllowDataSplit(3 + (lg_info.type == GROUP_3D ? 1 : 0),
                                        lg_info.type))) {
        // make min_wslice >= 256 to avoid gdma performance drop
        max_wsecs = std::min(max_wsecs, w);
        // avoid A2 chip ddr interleave
        if (w * dtype_bytes == 512 && module::isBM1688()) {
          max_wsecs = 1;
        }
        total_secs *= w;
      } else {
        max_wsecs = 1;
      }
      if (min_total_secs > total_secs) {
        min_total_secs = total_secs;
      }
    }
    vec_op_hwsecs.push_back(std::make_pair(op, min_total_secs));
  }

  shape_secs_t ret_secs;
  ret_secs.nsecs = max_nsecs;
  ret_secs.hsecs = max_hsecs;
  ret_secs.dsecs = max_dsecs;
  ret_secs.wsecs = max_wsecs;
  ret_secs.csecs = max_csecs;
  return ret_secs;
}

void update_multi_core_secs(const shape_secs_t max_shape_secs,
                            shape_secs_t &shape_secs) {
  auto core_num = module::getCoreNum();
  int64_t secs = shape_secs.nsecs * shape_secs.csecs * shape_secs.hsecs;
  int64_t max_secs =
      max_shape_secs.nsecs * max_shape_secs.csecs * max_shape_secs.hsecs;
  if (max_secs < core_num || secs >= core_num)
    return;

  shape_secs.nsecs = max_shape_secs.nsecs;
  secs = core_num / shape_secs.nsecs;
  if (shape_secs.csecs < secs && max_shape_secs.csecs >= secs) {
    shape_secs.csecs = secs;
  } else if (shape_secs.csecs < secs && max_shape_secs.csecs >= secs / 2) {
    shape_secs.csecs = secs / 2;
  }

  secs /= shape_secs.csecs;
  if (shape_secs.hsecs < secs && max_shape_secs.hsecs >= secs) {
    shape_secs.hsecs = secs;
  } else if (shape_secs.hsecs < secs && max_shape_secs.hsecs >= secs / 2) {
    shape_secs.hsecs = secs / 2;
  }
}

bool init_group_data_secs(const LgInfo &lg_info, shape_secs_t &shape_secs,
                          std::vector<std::pair<Value, int64_t>> &value_size,
                          const LgOptions &options) {
  if (lg_info.group_ops.size() == 1 && false == options.group_by_cores) {
    return true;
  }

  std::vector<std::pair<Operation *, int>> vec_op_hwsecs;
  shape_secs_t max_shape_secs = get_group_max_secs(lg_info, vec_op_hwsecs);

  int64_t in_n, in_c, in_d, in_h, in_w;
  int64_t out_n, out_c, out_d, out_h, out_w;
  for (auto op : lg_info.group_ops) {
    auto ins = get_input_values(op);
    auto outs = get_output_values(op);
    module::getNCDHW(ins[0], in_n, in_c, in_d, in_h, in_w, lg_info.type);
    module::getNCDHW(outs[0], out_n, out_c, out_d, out_h, out_w, lg_info.type);
    int64_t in0_lmem_bytes =
        Arch::get_tensor_lmem_bytes(ins[0], in_n, in_c, in_d, in_h, in_w);
    int64_t out0_lmem_bytes =
        Arch::get_tensor_lmem_bytes(outs[0], out_n, out_c, out_d, out_h, out_w);

    int64_t total_size = in0_lmem_bytes + out0_lmem_bytes;
    auto lg_op = cast<LocalGenInterface>(op);
    total_size += lg_op.getBufferSize(in0_lmem_bytes, out0_lmem_bytes, in_n,
                                      in_c, in_h, in_d, in_w, out_n, out_c,
                                      out_h, out_d, out_w, lg_info.type);
    for (size_t i = 1; i < ins.size(); ++i) {
      if ((module::isTrain() && !isa<tpu::AddOp, tpu::ConcatOp>(op)) ||
          module::isWeight(ins[i])) {
        bool eu_align = is_eu_align(ins[i]);
        int w_size =
            Arch::get_weight_lmem_bytes(ins[i], lg_info.type, eu_align);
        total_size += w_size;
        value_size.push_back(std::make_pair(ins[i], (w_size + 63) / 64 * 64));
      } else {
        module::getNCDHW(ins[i], in_n, in_c, in_d, in_h, in_w, lg_info.type);
        total_size +=
            Arch::get_tensor_lmem_bytes(ins[i], in_n, in_c, in_d, in_h, in_w);
      }
    }
    for (size_t i = 1; i < outs.size(); ++i) {
      module::getNCDHW(outs[i], out_n, out_c, out_d, out_h, out_w,
                       lg_info.type);
      total_size += Arch::get_tensor_lmem_bytes(outs[i], out_n, out_c, out_d,
                                                out_h, out_w);
    }
    // Need consider different backends
    int64_t total_secs = ceiling_func(total_size, Arch::LMEM_BYTES);
    shape_secs.nsecs =
        std::max(std::min(total_secs, max_shape_secs.nsecs), shape_secs.nsecs);
    total_secs = ceiling_func(total_secs, shape_secs.nsecs);
    if (lg_info.type == GROUP_MM || lg_info.type == GROUP_SMALL_C) {
      if (total_secs > max_shape_secs.csecs) {
        shape_secs.csecs = max_shape_secs.csecs;
      } else {
        int64_t cslice_per_npu = max_shape_secs.csecs / total_secs;
        shape_secs.csecs =
            std::max(ceiling_func(max_shape_secs.csecs, cslice_per_npu),
                     shape_secs.csecs);
      }
      total_secs = ceiling_func(total_secs, shape_secs.csecs);
    }
    shape_secs.dsecs =
        std::max(std::min(total_secs, max_shape_secs.dsecs), shape_secs.dsecs);
    total_secs = ceiling_func(total_secs, shape_secs.dsecs);
    shape_secs.hsecs = std::max(total_secs, shape_secs.hsecs);
    if (shape_secs.hsecs > max_shape_secs.hsecs) {
      shape_secs.wsecs = ceiling_func(shape_secs.hsecs, max_shape_secs.hsecs);
      if (shape_secs.wsecs > max_shape_secs.wsecs) {
        // llvm::outs() << "fail at op:"<<module::getName(op).str()<<"\n";
        return false;
      }
      shape_secs.hsecs = max_shape_secs.hsecs;
    }
  }
  return true;
}

bool init_group_data_secs2(ilp_LgInfo &ilp_lg_info, shape_secs_t &shape_secs,
                           std::vector<std::pair<Value, int64_t>> &value_size,
                           Operation *&fail_op,
                           std::shared_ptr<dot_graph> dot_graph_log,
                           const LgOptions &options) {
  fail_op = nullptr;
  auto lg_info = ilp_lg_info._lgInfo;
  if (lg_info.group_ops.size() == 1 && false == options.group_by_cores) {
    return true;
  }
  std::vector<std::pair<Operation *, int>> vec_op_hwsecs;
  shape_secs_t max_shape_secs = get_group_max_secs(lg_info, vec_op_hwsecs);

  int64_t in_n, in_c, in_d, in_h, in_w;
  int64_t out_n, out_c, out_d, out_h, out_w;
  for (auto op : lg_info.group_ops) {
    if (!op)
      continue;
    // llvm::errs() << "init_group_data_secs2 for op:"
    //               << module::getName(op).str() << "\n";
    auto ins = get_input_values(op);
    auto outs = get_output_values(op);
    module::getNCDHW(ins[0], in_n, in_c, in_d, in_h, in_w, lg_info.type);
    module::getNCDHW(outs[0], out_n, out_c, out_d, out_h, out_w, lg_info.type);
    int64_t in0_lmem_bytes =
        Arch::get_tensor_lmem_bytes(ins[0], in_n, in_c, in_d, in_h, in_w);
    // llvm::errs() << "  in0_lmem_bytes:" << in0_lmem_bytes<< ", in_n:" <<in_n
    //              << ", in_c:" <<in_c<< ", in_h:" <<in_h<< ", in_w:" <<in_w<<
    //              "\n";
    int64_t out0_lmem_bytes =
        Arch::get_tensor_lmem_bytes(outs[0], out_n, out_c, out_d, out_h, out_w);
    // llvm::errs() << "  out0_lmem_bytes:" << out0_lmem_bytes<< ", out_n:"
    // <<out_n
    //              << ", out_c:" <<out_c<< ", out_h:" <<out_h<< ", out_w:"
    //              <<out_w<< "\n";

    int64_t total_size = in0_lmem_bytes + out0_lmem_bytes;
    auto lg_op = cast<LocalGenInterface>(op);
    int64_t buffer_size = lg_op.getBufferSize(
        in0_lmem_bytes, out0_lmem_bytes, in_n, in_c, in_h, in_d, in_w, out_n,
        out_c, out_h, out_d, out_w, lg_info.type);
    total_size += buffer_size;
    // llvm::errs() << "  buffer_size:" << buffer_size<< ", total_size:"
    // <<total_size<< "\n";
    int64_t non_weight_size = Arch::LMEM_BYTES;
    for (size_t i = 1; i < ins.size(); ++i) {
      if ((module::isTrain() &&
           !isa<tpu::AddOp, tpu::SubOp, tpu::MulOp, tpu::DivOp, tpu::MinOp,
                tpu::MaxOp, tpu::ConcatOp>(op)) ||
          module::isWeight(ins[i])) {
        bool eu_align = is_eu_align(ins[i]);
        int w_size =
            Arch::get_weight_lmem_bytes(ins[i], lg_info.type, eu_align);
        // llvm::errs() << "  w_size:" << w_size<< "\n";
        // total_size += w_size;
        if (eu_align) {
          non_weight_size -= w_size;
        }
        value_size.push_back(std::make_pair(ins[i], (w_size + 63) / 64 * 64));
      } else {
        module::getNCDHW(ins[i], in_n, in_c, in_d, in_h, in_w, lg_info.type);
        total_size +=
            Arch::get_tensor_lmem_bytes(ins[i], in_n, in_c, in_d, in_h, in_w);
        // llvm::errs() << "  total_size:" << total_size<< "\n";
      }
    }
    for (size_t i = 1; i < outs.size(); ++i) {
      module::getNCDHW(outs[i], out_n, out_c, out_d, out_h, out_w,
                       lg_info.type);
      total_size += Arch::get_tensor_lmem_bytes(outs[i], out_n, out_c, out_d,
                                                out_h, out_w);
      // llvm::errs() << "  total_size:" << total_size<< "\n";
    }

    // Need consider different backends
    // Let's say the weights are also divided. In fact, the weights cannot be
    // divided
    // int64_t total_secs = ceiling_func(total_size, Arch::LMEM_BYTES);
    if (non_weight_size <= 0) {
      fail_op = op;
      return false;
    }
    int64_t total_secs = ceiling_func(total_size, non_weight_size);
    // llvm::errs() << "  total_secs:" << total_secs<< ", non_weight_size:"
    // <<non_weight_size<< "\n";
    shape_secs.nsecs =
        std::max(std::min(total_secs, max_shape_secs.nsecs), shape_secs.nsecs);
    total_secs = ceiling_func(total_secs, shape_secs.nsecs);
    if (lg_info.type == GROUP_MM || lg_info.type == GROUP_SMALL_C ||
        lg_info.type == GROUP_NORMAL) {
      if (total_secs > max_shape_secs.csecs) {
        shape_secs.csecs = max_shape_secs.csecs;
      } else {
        int64_t cslice_per_npu = max_shape_secs.csecs / total_secs;
        shape_secs.csecs =
            std::max(ceiling_func(max_shape_secs.csecs, cslice_per_npu),
                     shape_secs.csecs);
      }
      total_secs = ceiling_func(total_secs, shape_secs.csecs);
    }
    shape_secs.dsecs =
        std::max(std::min(total_secs, max_shape_secs.dsecs), shape_secs.dsecs);
    total_secs = ceiling_func(total_secs, shape_secs.dsecs);
    shape_secs.hsecs = std::max(total_secs, shape_secs.hsecs);
    auto name = module::getName(op).str();
    dot_graph_log->add_node_label(name + "_ori",
                                  "hsecs: " + std::to_string(total_secs));
    if (shape_secs.hsecs > max_shape_secs.hsecs) {
      shape_secs.wsecs = ceiling_func(shape_secs.hsecs, max_shape_secs.hsecs);
      if (shape_secs.wsecs > max_shape_secs.wsecs) {
        llvm::errs() << "init_group_data_secs2 fail at op:"
                     << module::getName(op).str() << "\n";
        return false;
      }
      shape_secs.hsecs = max_shape_secs.hsecs;
    }
  }
  return true;
}

static bool inc_slice_num(int &n_slice, int &c_slice, int &d_slice,
                          int &h_slice, int &w_slice,
                          const shape_secs_t &max_shape_secs) {
  if (n_slice < max_shape_secs.nsecs) {
    n_slice++;
  } else if (c_slice < max_shape_secs.csecs) {
    c_slice++;
  } else if (d_slice < max_shape_secs.dsecs) {
    d_slice++;
  } else if (h_slice < max_shape_secs.hsecs) {
    h_slice++;
  } else if (w_slice < max_shape_secs.wsecs) {
    w_slice++;
  } else {
    return false;
  }
  return true;
}

void get_op_cut_sec_num(
    ilp_LgInfo &ilp_lg_info,
    std::vector<std::pair<Operation *, int>> &vec_op_cut_secs) {
  auto lg_info = ilp_lg_info._lgInfo;
  shape_secs_t max_shape_secs;
  int64_t in_n, in_c, in_d, in_h, in_w, out_n, out_c, out_d, out_h, out_w;
  for (auto op : lg_info.group_ops) {
    if (!op)
      continue;
    int n_slice = 1, c_slice = 1, d_slice = 1, h_slice = 1, w_slice = 1;
    auto ins = get_input_values(op);
    auto outs = get_output_values(op);
    module::getNCDHW(ins[0], in_n, in_c, in_d, in_h, in_w, lg_info.type);
    module::getNCDHW(outs[0], out_n, out_c, out_d, out_h, out_w, lg_info.type);
    max_shape_secs.nsecs = out_n;
    max_shape_secs.csecs = out_c;
    max_shape_secs.dsecs = out_d;
    max_shape_secs.hsecs = out_h;
    max_shape_secs.wsecs = out_w;
    do {
      if (!inc_slice_num(n_slice, c_slice, d_slice, h_slice, w_slice,
                         max_shape_secs)) {
        break;
      }
      int new_in_n = ceiling_func(in_n, n_slice);
      int new_in_c = ceiling_func(in_c, c_slice);
      int new_in_d = ceiling_func(in_d, d_slice);
      int new_in_h = ceiling_func(in_h, h_slice);
      int new_in_w = ceiling_func(in_w, w_slice);
      int64_t in0_lmem_bytes = Arch::get_tensor_lmem_bytes(
          ins[0], new_in_n, new_in_c, new_in_d, new_in_h, new_in_w);
      int new_out_n = ceiling_func(out_n, n_slice);
      int new_out_c = ceiling_func(out_c, c_slice);
      int new_out_d = ceiling_func(out_d, d_slice);
      int new_out_h = ceiling_func(out_h, h_slice);
      int new_out_w = ceiling_func(out_w, w_slice);
      int64_t out0_lmem_bytes = Arch::get_tensor_lmem_bytes(
          outs[0], new_out_n, new_out_c, new_out_d, new_out_h, new_out_w);
      int64_t total_size = in0_lmem_bytes + out0_lmem_bytes;
      auto lg_op = cast<LocalGenInterface>(op);
      int64_t buffer_size = lg_op.getBufferSize(
          in0_lmem_bytes, out0_lmem_bytes, new_in_n, new_in_c, new_in_h,
          new_in_d, new_in_w, new_out_n, new_out_c, new_out_h, new_out_d,
          new_out_w, lg_info.type);
      total_size += buffer_size;
      int64_t non_weight_size = Arch::LMEM_BYTES;
      for (size_t i = 1; i < ins.size(); ++i) {
        if (is_value_dont_split(ins[i])) {
          int w_size = Arch::get_weight_lmem_bytes(ins[i], lg_info.type);
          non_weight_size -= w_size;
        } else {
          module::getNCDHW(ins[i], in_n, in_c, in_d, in_h, in_w, lg_info.type);
          total_size +=
              Arch::get_tensor_lmem_bytes(ins[i], in_n, in_c, in_d, in_h, in_w);
        }
      }
      for (size_t i = 1; i < outs.size(); ++i) {
        module::getNCDHW(outs[i], out_n, out_c, out_d, out_h, out_w,
                         lg_info.type);
        total_size += Arch::get_tensor_lmem_bytes(outs[i], out_n, out_c, out_d,
                                                  out_h, out_w);
      }
      if (total_size <= Arch::LMEM_BYTES) {
        break;
      }
    } while (true);
    vec_op_cut_secs.push_back(
        std::make_pair(op, n_slice * c_slice * d_slice * h_slice * w_slice));
  }
}

int64_t get_split_max_secs(BasicTimeStepPtr time_step) {
  int64_t timestep_num = time_step->get_timestep_num();
  if (timestep_num == 0) {
    return 0;
  }
  std::vector<int64_t> lmem_req(timestep_num, 0);
  const MemBuff &lmem_buffer = time_step->get_lmem_buffer();

  auto update_lmem_req = [&lmem_req, &timestep_num](int64_t start_ts,
                                                    int64_t end_ts,
                                                    int64_t lmem_size) {
    if (start_ts <= end_ts) {
      for (int64_t ts = start_ts; ts <= end_ts; ++ts) {
        lmem_req[ts] += lmem_size;
      }
    } else {
      for (int64_t ts = 0; ts <= end_ts; ++ts) {
        lmem_req[ts] += lmem_size;
      }
      for (int64_t ts = start_ts; ts <= timestep_num - 1; ++ts) {
        lmem_req[ts] += lmem_size;
      }
    }
  };

  for (auto iter = lmem_buffer.begin(); iter != lmem_buffer.end(); ++iter) {
    int64_t start_ts = iter->second.start_ts;
    int64_t end_ts = iter->second.end_ts;
    update_lmem_req(start_ts, end_ts, (iter->second).size);
  }

  std::stable_sort(lmem_req.begin(), lmem_req.end(), std::greater<int64_t>());
  return ceiling_func(lmem_req[0], Arch::LMEM_BYTES);
}

void update_tensor_infos(const LgInfo &lg_info, TensorInfo &tensor_infos,
                         const shape_secs_t &shape_secs, int speical_pattern) {
  for (auto &iter : tensor_infos) {
    auto v = iter.first;
    iter.second.use_3ic_opt = use_3ic(v);
    iter.second.eu_align = is_eu_align(v);
    iter.second.need_bcast = need_bcast(v);
  }

  if (speical_pattern > 0) {
    return;
  }

  tensor_info_t ti(TIMESTEP_LOAD);
  int64_t n, c, d, h, w;
  for (auto op : lg_info.group_ops) {
    if (!op) {
      continue;
    }
    auto ins = get_input_values(op);
    for (auto in : ins) {
      if (auto src_op = dyn_cast_or_null<top::WeightOp>(in.getDefiningOp())) {
        bool allow_split = false;
        if (src_op.getAllowSplitAttr() != nullptr) {
          allow_split = true;
        }
        if (allow_split == false) {
          ti.eu_align = is_eu_align(in);
          ti.need_bcast = need_bcast(in);
          module::getNCDHW(in, n, c, d, h, w, lg_info.type);
          ti.slice_info.n.clear();
          ti.slice_info.c.clear();
          ti.slice_info.d.clear();
          ti.slice_info.h.clear();
          ti.slice_info.w.clear();
          ti.slice_info.n.push_back(std::make_pair((int64_t)0, (int64_t)n));
          ti.slice_info.c.push_back(std::make_pair((int64_t)0, (int64_t)c));
          ti.slice_info.d.push_back(std::make_pair((int64_t)0, (int64_t)d));
          ti.slice_info.h.push_back(std::make_pair((int64_t)0, (int64_t)h));
          ti.slice_info.w.push_back(std::make_pair((int64_t)0, (int64_t)w));
          tensor_infos[in] = ti;
        }
      }
    }
  }
}

bool can_split_w(const LgInfo &lg_info, int64_t dhw_secs, int64_t height_min,
                 int64_t wsecs) {
  if (dhw_secs < height_min && wsecs > 1) {
    for (auto out : lg_info.group_outs) {
      int64_t n, c, d, h, w;
      module::getNCDHW(out, n, c, d, h, w, lg_info.type);
      int dtype_size = module::getDtypeSize(out);
      if (ceiling_func(w, wsecs) * dtype_size < 256) {
        return false;
      }
    }
  }
  return true;
}

// make sure group secs can be devided by num_core
static void force_group_by_cores(shape_secs_t &shape_secs,
                                 const shape_secs_t &max_shape_secs) {
  auto num_cores = module::getCoreNum();
  if (num_cores < 2) {
    return;
  }
  auto pre_secs = shape_secs.nsecs * shape_secs.csecs * shape_secs.dsecs;
  if (pre_secs * shape_secs.hsecs % num_cores == 0) {
    return;
  }
  for (int i = 1; i < num_cores; i++) {
    if ((shape_secs.hsecs + i) > max_shape_secs.hsecs) {
      return;
    }
    if (pre_secs * (shape_secs.hsecs + i) % num_cores == 0) {
      shape_secs.hsecs += i;
      return;
    }
  }
  return;
}

void assign_dhwsecs(const LgInfo &lg_info, shape_secs_t &shape_secs,
                    int64_t &dhw_secs, const shape_secs_t &max_shape_secs,
                    const LgOptions &options) {
  shape_secs.dsecs = 1;
  shape_secs.hsecs = dhw_secs;
  shape_secs.wsecs = 1;
  ValueSet group_out_tensors;
  for (auto op : lg_info.group_ops) {
    auto outs = get_output_values(op);
    group_out_tensors.insert(outs.begin(), outs.end());
  }
  if (max_shape_secs.dsecs == 1) {
    shape_secs.dsecs = 1;
    shape_secs.hsecs = dhw_secs;
    shape_secs.wsecs = 1;
    // split height and width
    float h_len = 0.f, w_len = 0.f;
    for (auto out : group_out_tensors) {
      if (out.use_empty()) {
        continue;
      }
      int64_t n, c, d, h, w;
      module::getNCDHW(out, n, c, d, h, w, lg_info.type);
      h_len += (float)h;
      w_len += (float)w;
    }

    float min_len = __FLT_MAX__;
    for (int64_t i = max_shape_secs.hsecs; i > 0; --i) {
      int64_t hsecs = i;
      int64_t wsecs = ceiling_func(dhw_secs, i);

      int cur_len = (hsecs - 1) + (wsecs - 1) * (h_len / w_len);
      bool split_w =
          can_split_w(lg_info, dhw_secs, max_shape_secs.hsecs, wsecs);

      if (cur_len < min_len && split_w && wsecs <= max_shape_secs.wsecs) {
        min_len = cur_len;
        shape_secs.hsecs = hsecs;
        shape_secs.wsecs = wsecs;
      }
    }
  } else {
    // split depth and height
    if (module::isBM1688() || module::isSG2380() || module::isMARS3() ||
        module::isSGTPUV8()) {
      float d_len = 0.f, h_len = 0.f;
      for (auto out : group_out_tensors) {
        int64_t n, c, d, h, w;
        module::getNCDHW(out, n, c, d, h, w, lg_info.type);
        d_len += (float)d;
        h_len += (float)h;
      }
      float min_len = __FLT_MAX__;
      for (int64_t i = max_shape_secs.dsecs; i > 0; --i) {
        int64_t dsecs = i;
        int64_t hsecs = ceiling_func(dhw_secs, i);

        int cur_len = (dsecs - 1) + (hsecs - 1) * (d_len / h_len);
        if (cur_len < min_len) {
          min_len = cur_len;
          shape_secs.dsecs = dsecs;
          shape_secs.hsecs = hsecs;
        }
      }
    } else {
      /// if split h or w, gdma band width may be lowered due to ddr 4k channel
      /// interleave
      shape_secs.dsecs = std::min(max_shape_secs.dsecs, dhw_secs);
      shape_secs.hsecs = ceiling_func(dhw_secs, shape_secs.dsecs);
    }
    // d split is max but h max still not enough, split w
    if (shape_secs.hsecs > max_shape_secs.hsecs) {
      shape_secs.wsecs = ceiling_func(shape_secs.hsecs, max_shape_secs.hsecs);
      shape_secs.hsecs = max_shape_secs.hsecs;
    }
  }
  if (options.group_by_cores) {
    force_group_by_cores(shape_secs, max_shape_secs);
  }
  dhw_secs = shape_secs.dsecs * shape_secs.hsecs * shape_secs.wsecs;
}

bool update_data_split(BasicTimeStepPtr time_step, const LgInfo &lg_info,
                       shape_secs_t &shape_secs, const LgOptions &options) {
  shape_secs.clear();
  bool status = false;
  auto &tensor_infos = time_step->get_tensor_infos();
  std::vector<std::pair<Operation *, int>> vec_op_hwsecs;
  shape_secs_t max_shape_secs = get_group_max_secs(lg_info, vec_op_hwsecs);
  int64_t cdhw_secs = 0;
  for (int64_t nsec = 1; nsec <= max_shape_secs.nsecs; ++nsec) {
    shape_secs.nsecs = nsec;
    tensor_infos.clear();
    if (stripe_mine_max_slice(lg_info, shape_secs, tensor_infos, options) ==
        false) {
      return false;
    }
    time_step->update_all_mem_buffer_size(lg_info);

    cdhw_secs = get_split_max_secs(time_step);
    if (cdhw_secs == 0) {
      return false;
    }
    // update csecs
    shape_secs.csecs =
        std::max(shape_secs.csecs, std::min(max_shape_secs.csecs, cdhw_secs));
    // update d/h/w secs
    int64_t dhw_secs = ceiling_func(cdhw_secs, shape_secs.csecs);
    if (dhw_secs > 1) {
      if (shape_secs.nsecs == max_shape_secs.nsecs) {
        assign_dhwsecs(lg_info, shape_secs, dhw_secs, max_shape_secs, options);
      } else {
        shape_secs.clear();
        continue;
      }
    }
    if (shape_secs.dsecs <= max_shape_secs.dsecs &&
        shape_secs.hsecs <= max_shape_secs.hsecs &&
        shape_secs.wsecs <= max_shape_secs.wsecs) {
      status = true;
      break;
    }
  }

  update_tensor_infos(lg_info, tensor_infos, shape_secs);
  auto status_str = status ? "success" : "failed";
  GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
    llvm::dbgs()
        << DEBUGGER_DEFAULT_INFO(
               "updata_data_split_result", status_str,
               "try to find more appropriate shape_secs than the initial one")
        << "\n";
  });
  return status;
}

bool strip_back_judge(Value v, const LgInfo &lg_info,
                      const std::multiset<Operation *> &op_set,
                      const std::set<Value, value_compare> &out_tensor_set) {
  auto users = v.getUsers();
  bool res = true;
  bool has_outer_group_user = false;
  for (auto op : users) {
    if (std::find(lg_info.group_ops.begin(), lg_info.group_ops.end(), op) !=
        lg_info.group_ops.end()) {
      if (op_set.count(op) == 0) {
        return false;
      }
    } else {
      has_outer_group_user = true;
    }
  }

  if (has_outer_group_user) {
    res = out_tensor_set.find(v) != out_tensor_set.end();
  }
  return res;
}

bool strip_back_judge2(Value v, const LgInfo &lg_info,
                       const std::multiset<Operation *> &op_set,
                       const std::set<Value, value_compare> &out_tensor_set) {
  auto users = v.getUsers();
  bool res = true;
  // bool has_outer_group_user = false;
  for (auto op : users) {
    if (std::find(lg_info.group_ops.begin(), lg_info.group_ops.end(), op) !=
        lg_info.group_ops.end()) {
      if (op_set.count(op) == 0) {
        return false;
      }
    } else {
      // has_outer_group_user = true;
    }
  }

  // if (has_outer_group_user) {
  //   res = out_tensor_set.find(v) != out_tensor_set.end();
  // }
  return res;
}

inline bool is_same_slice(const slice_pair_t &a, const slice_pair_t &b) {
  return (a.first == b.first) && (a.second == b.second);
}

bool is_same_slice_info(const slice_info_t &si0, const slice_info_t &si1) {
  if (si0.n.size() != si1.n.size() || si0.c.size() != si1.c.size() ||
      si0.h.size() != si1.h.size() || si0.d.size() != si1.d.size() ||
      si0.w.size() != si1.w.size()) {
    return false;
  }
  // check n
  for (size_t i = 0; i < si0.n.size(); ++i) {
    if (false == is_same_slice(si0.n[i], si1.n[i])) {
      return false;
    }
  }
  // check c
  for (size_t i = 0; i < si0.c.size(); ++i) {
    if (false == is_same_slice(si0.c[i], si1.c[i])) {
      return false;
    }
  }
  // check h
  for (size_t i = 0; i < si0.h.size(); ++i) {
    if (false == is_same_slice(si0.h[i], si1.h[i])) {
      return false;
    }
  }
  // check d
  for (size_t i = 0; i < si0.d.size(); ++i) {
    if (false == is_same_slice(si0.d[i], si1.d[i])) {
      return false;
    }
  }
  // check w
  for (size_t i = 0; i < si0.w.size(); ++i) {
    if (false == is_same_slice(si0.w[i], si1.w[i])) {
      return false;
    }
  }
  // for (auto it : llvm::zip(si0.n, si1.n)) {
  //   if (false == is_same_slice(std::get<0>(it), std::get<1>(it))) {
  //     return false;
  //   }
  // }
  // // check h
  // for (auto it : llvm::zip(si0.h, si1.h)) {
  //   if (false == is_same_slice(std::get<0>(it), std::get<1>(it))) {
  //     return false;
  //   }
  // }
  return true;
}

bool is_broadcast_binary(Operation *op, Value in) {
  if (!isa<tpu::AddOp, tpu::SubOp, tpu::MulOp, tpu::DivOp, tpu::MaxOp,
           tpu::MinOp, tpu::BinaryShiftOp>(op)) {
    return false;
  }
  auto other = in == op->getOperand(0) ? op->getOperand(1) : op->getOperand(0);
  auto in_shape = in.getType().cast<RankedTensorType>().getShape();
  auto other_shape = other.getType().cast<RankedTensorType>().getShape();
  if (in_shape.size() != other_shape.size()) {
    return false;
  }
  for (int i = 0; i < in_shape.size(); ++i) {
    if (in_shape[i] != other_shape[i] && in_shape[i] == 1) {
      return true;
    }
  }
  return false;
}

// if no transpose, split left matrix rows,
// if left/right are both transposed, split right matrix rows
bool is_matmul_right_tensor(Operation *op, Value v) {
  bool res = false;
  if (auto matmul_op = dyn_cast<tpu::MatMulOp>(op)) {
    bool left_trans = matmul_op.getLeftTranspose();
    bool right_trans = matmul_op.getRightTranspose();
    res = (v == matmul_op.getRight());
    if (left_trans && right_trans) {
      res = (v == matmul_op.getInput());
    }
  }
  return res;
}

bool is_broadcast_rope_with_permute_optimize(Operation *op) {
  if (auto rope_op = dyn_cast<tpu::RopeOp>(op)) {
    if (!rope_op.getIsPermuteOptimize()) {
      return false;
    }
    auto rope_input1 = rope_op.getInput1();
    auto rope_input2 = rope_op.getInput2();
    auto rope_input3 = rope_op.getInput3();
    auto input1_shape =
        rope_input1.getType().cast<RankedTensorType>().getShape();
    auto input2_shape =
        rope_input2.getType().cast<RankedTensorType>().getShape();
    auto input3_shape =
        rope_input3.getType().cast<RankedTensorType>().getShape();
    if (input1_shape.size() != 4 || input2_shape.size() != 4 ||
        input3_shape.size() != 4) {
      return false;
    }
    if (input1_shape[1] != input2_shape[1] ||
        input1_shape[1] != input3_shape[1]) {
      return false;
    }
    if (input1_shape[3] != input2_shape[3] ||
        input1_shape[3] != input3_shape[3]) {
      return false;
    }
    if ((input1_shape[0] != input2_shape[0] && input2_shape[0] != 1) ||
        (input1_shape[0] != input3_shape[0] && input3_shape[0] != 1)) {
      return false;
    }
    if ((input1_shape[2] != input2_shape[2] && input2_shape[2] != 1) ||
        (input1_shape[2] != input3_shape[2] && input3_shape[2] != 1)) {
      return false;
    }
    return true;
  }
  return false;
}

bool is_attention_not_input_tensor(Operation *op, Value v) {
  bool res = false;
  if (auto attention_op = dyn_cast<tpu::AttentionOp>(op)) {
    res = (v != attention_op.getInput());
  }
  return res;
}

void slice_distributor(std::vector<slice_pair_t> &slice_pairs,
                       int64_t vec_length, int64_t secs) {
  slice_pairs.clear();
  int64_t per_sec_base = vec_length / secs; // 每个分片的基本大小
  int64_t extra = vec_length % secs; // 需要额外分配一个单位的分片数量

  int64_t start_idx = 0; // 当前分片的起始索引
  for (int64_t i = 0; i < secs; ++i) {
    int64_t current_slice =
        per_sec_base + (i < extra ? 1 : 0); // 当前分片的大小
    if (current_slice > 0) {
      slice_pairs.emplace_back(slice_pair_t(start_idx, current_slice));
      start_idx += current_slice; // 更新下一个分片的起始索引
    }
  }
}

slice_info_t get_out_slice_info(const shape_secs_t &shape_secs, int64_t n,
                                int64_t c, int64_t h, int64_t d, int64_t w,
                                int64_t bitwidth) {
  slice_info_t slice_info;
  int64_t secs, idx, slice, step;
  // n slice info
  secs = shape_secs.nsecs;
  int64_t n_align = 32 / bitwidth;
  if (Arch::ALIGN_4N && n_align != 1) {
    step = align_up(ceiling_func(n, secs), n_align);
    for (int64_t i = 0; i < secs; ++i) {
      idx = i == 0 ? 0 : idx + slice;
      slice = (n - idx) > step ? step : (n - idx);
      slice_info.n.emplace_back(slice_pair_t(idx, slice));
    }
  } else {
    slice_distributor(slice_info.n, n, shape_secs.nsecs);
  }
  // c slice info
  auto npu_num = Arch::NPU_NUM;
  secs = shape_secs.csecs;
  // if (secs * npu_num > c){
  //    npu_num = ceiling_func(c, secs);
  // }
  int64_t c_per_npu = ceiling_func(c, npu_num);
  int64_t c_per_npu_div_secs = c_per_npu / secs;
  int64_t c_per_npu_mod_secs = c_per_npu % secs;
  for (int64_t i = 0; i < secs; ++i) {
    // 计算每一步的步长和索引
    bool extra = c_per_npu_mod_secs > i;
    int64_t step = (c_per_npu_div_secs + extra) * npu_num;
    int64_t idx =
        (c_per_npu_div_secs * i + (extra ? i : c_per_npu_mod_secs)) * npu_num;

    // 计算切片大小
    int64_t slice = std::min(step, c - idx);
    assert(idx < c);
    slice_info.c.emplace_back(slice_pair_t(idx, slice));
  }
  // h slice_info
  slice_distributor(slice_info.h, h, shape_secs.hsecs);
  // d slice_info
  slice_distributor(slice_info.d, d, shape_secs.dsecs);
  // w slice_info
  slice_distributor(slice_info.w, w, shape_secs.wsecs);
  return slice_info;
}

bool get_backward_slice_info(slice_info_t &in_si, const slice_info_t &out_si,
                             Operation *op, Value in,
                             const shape_secs_t &shape_secs,
                             group_type_t group_type, bool &hold_in_lmem,
                             bool is_group_in) {
  int64_t n, c, d, h, w;
  module::getNCDHW(in, n, c, d, h, w, group_type);
  auto lg_op = cast<LocalGenInterface>(op);
  bool is_broadcast_tensor = is_broadcast_binary(op, in);
  bool is_right_matrix = is_matmul_right_tensor(op, in);
  bool is_no_input_attention = is_attention_not_input_tensor(op, in);
  bool is_broadcast_rope = is_broadcast_rope_with_permute_optimize(op);

  int64_t idx = 0, slice = 0;
  if (shape_secs.nsecs == 1) {
    in_si.n.emplace_back(slice_pair_t(0, n));
  } else {
    for (auto &s : out_si.n) {
      auto ret = lg_op.BackwardN(idx, slice, s.first, s.second);
      if ((is_broadcast_tensor || is_right_matrix || is_broadcast_rope) &&
          n == 1) {
        idx = 0;
        slice = 1;
      } else {
        if (failed(ret) || slice == 0) {
          LLVM_DEBUG(llvm::dbgs() << "BackwardN fail, at op:"
                                  << module::getName(op).str() << "\n";);
          return false;
        }
      }
      in_si.n.emplace_back(slice_pair_t(idx, slice));
    }
  }

  if (shape_secs.csecs == 1) {
    in_si.c.emplace_back(slice_pair_t(0, c));
  } else {
    for (auto &s : out_si.c) {
      auto ret = lg_op.BackwardC(idx, slice, s.first, s.second);
      if (is_broadcast_tensor && c == 1) {
        idx = 0;
        slice = 1;
      } else if (is_right_matrix) {
        idx = 0;
        slice = c;
        in_si.c.emplace_back(slice_pair_t(idx, slice));
        if (is_group_in && in_si.n.size() == 1) {
          hold_in_lmem = true;
          break;
        } else {
          hold_in_lmem = false;
          continue;
        }
      } else if (is_no_input_attention) {
        idx = 0;
        slice = c;
        in_si.c.emplace_back(slice_pair_t(idx, slice));
      } else {
        if (failed(ret) || slice == 0) {
          // llvm::outs() << "BackwardC fail, at
          // op:"<<module::getName(op).str()<<"\n";
          return false;
        }
      }
      in_si.c.emplace_back(slice_pair_t(idx, slice));
    }
  }

  int64_t pre_end_idx = 0;
  idx = slice = 0;
  if (shape_secs.dsecs == 1) {
    in_si.d.emplace_back(slice_pair_t(0, d));
  } else {
    for (int i = 0; i < out_si.d.size(); i++) {
      auto &s = out_si.d[i];
      auto ret = lg_op.BackwardD(idx, slice, s.first, s.second);
      if (is_broadcast_tensor && d == 1) {
        idx = 0;
        slice = 1;
      } else {
        bool end_reached = idx + slice == pre_end_idx;
        if (failed(ret) || slice == 0 || (idx == 0 && i > 0) || end_reached) {
          LLVM_DEBUG(llvm::dbgs() << "BackwardD fail, at op:"
                                  << module::getName(op).str() << "\n";);
          return false;
        }
      }
      pre_end_idx = idx + slice;
      in_si.d.emplace_back(slice_pair_t(idx, slice));
    }
  }

  pre_end_idx = 0;
  idx = slice = 0;
  if (shape_secs.hsecs == 1) {
    in_si.h.emplace_back(slice_pair_t(0, h));
  } else {
    // llvm::errs()  << "BackwardH at op:" << module::getName(op).str() <<",
    // type:"<<op->getName()<< "\n";
    for (int i = 0; i < out_si.h.size(); i++) {
      auto &s = out_si.h[i];
      auto ret = lg_op.BackwardH(idx, slice, s.first, s.second);
      // llvm::errs()  <<"  i:"<<i<< ", out_idx:"<<s.first<< ", out_slice:"
      //               <<s.second<< " >>> in_idx:"<<idx<< ", in_slice:"<<slice
      //               << "\n";
      if ((is_right_matrix || is_broadcast_tensor || is_broadcast_rope) &&
          h == 1) {
        idx = 0;
        slice = 1;
      } else {
        bool end_reached = idx + slice == pre_end_idx;
        // TMP
        // if (failed(ret) || slice == 0 ){
        if (failed(ret) || slice == 0 || (idx == 0 && i > 0) || end_reached) {
          LLVM_DEBUG(llvm::dbgs()
                         << "BackwardH fail, ret:"
                         << (failed(ret) ? "failed" : "success")
                         << ", end_reached:" << end_reached << ", i:" << i
                         << ", idx:" << idx << ", slice:" << slice
                         << " at op:" << module::getName(op).str() << "\n";);
          // for debug
          // llvm::dbgs()
          //       << "BackwardH fail, ret:"
          //       << (failed(ret) ? "failed" : "success")
          //       << ", end_reached:" << end_reached << ", i:" << i
          //       << ", idx:" << idx << ", slice:" << slice
          //       << " at op:" << module::getName(op).str() << "\n";
          // for(auto it:out_si.h){
          //   llvm::dbgs() << it.first <<", "<<it.second <<"\n";
          // }
          // for debug
          return false;
        }
      }
      pre_end_idx = idx + slice;
      in_si.h.emplace_back(slice_pair_t(idx, slice));
    }
  }

  pre_end_idx = 0;
  idx = slice = 0;
  if (shape_secs.wsecs == 1) {
    in_si.w.emplace_back(slice_pair_t(0, w));
  } else {
    for (int i = 0; i < out_si.w.size(); i++) {
      auto &s = out_si.w[i];
      auto ret = lg_op.BackwardW(idx, slice, s.first, s.second);
      if (is_broadcast_tensor && w == 1) {
        idx = 0;
        slice = 1;
      } else {
        bool end_reached = idx + slice == pre_end_idx;
        if (failed(ret) || slice == 0 || (idx == 0 && i > 0) || end_reached) {
          LLVM_DEBUG(llvm::dbgs() << "BackwardW fail, at op:"
                                  << module::getName(op).str() << "\n";);
          return false;
        }
      }
      pre_end_idx = idx + slice;
      in_si.w.emplace_back(slice_pair_t(idx, slice));
    }
  }
  return true;
}

bool get_backward_slice_info2(slice_info_t &in_si, const slice_info_t &out_si,
                              Operation *op, Value in,
                              const shape_secs_t &shape_secs,
                              group_type_t group_type, bool &hold_in_lmem,
                              bool is_group_in) {
  int64_t n, c, d, h, w;
  module::getNCDHW(in, n, c, d, h, w, group_type);
  auto lg_op = cast<LocalGenInterface>(op);
  bool is_broadcast_tensor = is_broadcast_binary(op, in);
  bool is_right_matrix = is_matmul_right_tensor(op, in);
  bool is_no_input_attention = is_attention_not_input_tensor(op, in);

  int64_t idx = 0, slice = 0;
  if (shape_secs.nsecs == 1) {
    in_si.n.emplace_back(slice_pair_t(0, n));
  } else {
    for (auto &s : out_si.n) {
      auto ret = lg_op.BackwardN(idx, slice, s.first, s.second);
      if (is_broadcast_tensor && n == 1) {
        idx = 0;
        slice = 1;
      } else {
        if (failed(ret) || slice == 0) {
          llvm::errs() << "BackwardN fail, at op:" << module::getName(op).str()
                       << "\n";
          return false;
        }
      }
      in_si.n.emplace_back(slice_pair_t(idx, slice));
    }
  }

  if (shape_secs.csecs == 1) {
    in_si.c.emplace_back(slice_pair_t(0, c));
  } else {
    for (auto &s : out_si.c) {
      auto ret = lg_op.BackwardC(idx, slice, s.first, s.second);
      if (is_broadcast_tensor && c == 1) {
        idx = 0;
        slice = 1;
      } else if (is_right_matrix) {
        idx = 0;
        slice = c;
        in_si.c.emplace_back(slice_pair_t(idx, slice));
        if (is_group_in) {
          hold_in_lmem = true;
          break;
        } else {
          hold_in_lmem = false;
          continue;
        }
      } else if (is_no_input_attention) {
        idx = 0;
        slice = c;
        in_si.c.emplace_back(slice_pair_t(idx, slice));
      } else {
        if (failed(ret) || slice == 0) {
          llvm::errs() << "BackwardC fail, at op:" << module::getName(op).str()
                       << "\n";
          return false;
        }
      }
      in_si.c.emplace_back(slice_pair_t(idx, slice));
    }
  }

  int64_t pre_end_idx = 0;
  idx = slice = 0;
  if (shape_secs.dsecs == 1) {
    in_si.d.emplace_back(slice_pair_t(0, d));
  } else {
    for (int i = 0; i < out_si.d.size(); i++) {
      auto &s = out_si.d[i];
      auto ret = lg_op.BackwardD(idx, slice, s.first, s.second);
      if (is_broadcast_tensor && d == 1) {
        idx = 0;
        slice = 1;
      } else {
        bool end_reached = idx + slice == pre_end_idx;
        if (failed(ret) || slice == 0 || (idx == 0 && i > 0) || end_reached) {
          LLVM_DEBUG(llvm::dbgs() << "BackwardD fail, at op:"
                                  << module::getName(op).str() << "\n";);
          llvm::errs() << "BackwardD fail, at op:" << module::getName(op).str()
                       << "\n";
          return false;
        }
      }
      pre_end_idx = idx + slice;
      in_si.d.emplace_back(slice_pair_t(idx, slice));
    }
  }

  pre_end_idx = 0;
  idx = slice = 0;
  if (shape_secs.hsecs == 1) {
    in_si.h.emplace_back(slice_pair_t(0, h));
  } else {
    // llvm::errs()  << "BackwardH at op:" << module::getName(op).str() <<",
    // type:"<<op->getName()<< "\n";
    for (int i = 0; i < out_si.h.size(); i++) {
      auto &s = out_si.h[i];
      auto ret = lg_op.BackwardH(idx, slice, s.first, s.second);
      // llvm::errs()  <<"  i:"<<i<< ", out_idx:"<<s.first<< ", out_slice:"
      //               <<s.second<< " >>> in_idx:"<<idx<< ", in_slice:"<<slice
      //               << "\n";
      if ((is_right_matrix || is_broadcast_tensor) && h == 1) {
        idx = 0;
        slice = 1;
      } else {
        if (failed(ret) || slice == 0) {
          LLVM_DEBUG(llvm::dbgs()
                         << "BackwardH fail, ret:"
                         << (failed(ret) ? "failed" : "success") << ", i:" << i
                         << ", idx:" << idx << ", slice:" << slice
                         << " at op:" << module::getName(op).str() << "\n";);
          llvm::errs() << "BackwardH fail, ret:"
                       << (failed(ret) ? "failed" : "success") << ", i:" << i
                       << ", idx:" << idx << ", slice:" << slice
                       << " at op:" << module::getName(op).str() << "\n";
          // for debug
          // llvm::dbgs()
          //       << "BackwardH fail, ret:"
          //       << (failed(ret) ? "failed" : "success")
          //       << ", end_reached:" << end_reached << ", i:" << i
          //       << ", idx:" << idx << ", slice:" << slice
          //       << " at op:" << module::getName(op).str() << "\n";
          // for(auto it:out_si.h){
          //   llvm::dbgs() << it.first <<", "<<it.second <<"\n";
          // }
          // for debug
          return false;
        }
      }
      pre_end_idx = idx + slice;
      in_si.h.emplace_back(slice_pair_t(idx, slice));
    }
  }

  pre_end_idx = 0;
  idx = slice = 0;
  if (shape_secs.wsecs == 1) {
    in_si.w.emplace_back(slice_pair_t(0, w));
  } else {
    for (int i = 0; i < out_si.w.size(); i++) {
      auto &s = out_si.w[i];
      auto ret = lg_op.BackwardW(idx, slice, s.first, s.second);
      if (is_broadcast_tensor && w == 1) {
        idx = 0;
        slice = 1;
      } else {
        bool end_reached = idx + slice == pre_end_idx;
        if (failed(ret) || slice == 0 || (idx == 0 && i > 0) || end_reached) {
          LLVM_DEBUG(llvm::dbgs() << "BackwardW fail, at op:"
                                  << module::getName(op).str() << "\n";);
          llvm::errs() << "BackwardW fail, at op:" << module::getName(op).str()
                       << "\n";
          return false;
        }
      }
      pre_end_idx = idx + slice;
      in_si.w.emplace_back(slice_pair_t(idx, slice));
    }
  }
  return true;
}
bool check_hsecs(Value value, slice_info_t &si, group_type_t group_type) {
  assert(si.h.size() > 0);
  int64_t n, c, d, h, w;
  module::getNCDHW(value, n, c, d, h, w, group_type);
  int64_t total_h = 0;
  for (auto &it : si.h) {
    total_h += it.second;
  }
  if (total_h * 2 > h * 3) { // h increase 1.5 times
    return false;
  }
  return true;
}

static bool backward_update_slice(
    const LgInfo &lg_info, const shape_secs_t &shape_secs, const Value &out,
    std::list<Value> &tensor_branchs, TensorInfo &tensor_infos,
    std::multiset<Operation *> &op_set, const ValueSet &out_tensor_set) {

  // Don't backward when this out tensor is the input of the group
  if (std::find(lg_info.group_ins.begin(), lg_info.group_ins.end(), out) !=
      lg_info.group_ins.end()) {
    // return check_hsecs(out, tensor_infos[out].slice_info, lg_info.type);
    return true;
  }
  auto op = out.getDefiningOp();
  if (isa<tpu::Conv2DOp>(op) && module::isBM1684Family()) {
    auto conv_attr = dyn_cast<tpu::Conv2DOp>(op).parseParam();
    if (conv_attr.use_3ic_optimize) {
      return false;
    }
  }
  op_set.insert(op);
  auto mode = getRunMode(op);

  slice_info_t &out_si = tensor_infos[out].slice_info;
  auto &group_ins = lg_info.group_ins;

  for (auto in : op->getOperands()) {
    auto pre_op = in.getDefiningOp();
    if (pre_op != nullptr && isa<top::NoneOp>(pre_op)) {
      continue;
    }
    if (auto weight_op = dyn_cast_or_null<top::WeightOp>(pre_op)) {
      if (weight_op.getAllowSplit() == std::nullopt) {
        continue;
      }
      bool allow_split = true;
      auto weight_op_allow_split_attr = weight_op.getAllowSplitAttr();
      auto num_dims = weight_op_allow_split_attr.size();
      auto allow_split_array = module::getI64Array(weight_op_allow_split_attr);
      for (int i = 0; i < num_dims; ++i) {
        if (allow_split_array->at(i) == 0)
          allow_split = false;
      }
      if (allow_split == false) {
        continue;
      }
    }
    slice_info_t si;
    bool hold_in_lmem = false;
    bool is_group_in =
        std::find(group_ins.begin(), group_ins.end(), in) != group_ins.end();
    auto ret = get_backward_slice_info(si, out_si, op, in, shape_secs,
                                       lg_info.type, hold_in_lmem, is_group_in);
    if (pre_op && module::isDynWeight(in)) {
      auto shape = module::getShape(in);
      si.n.clear();
      si.n.emplace_back(std::pair(0, shape[0]));
      si.c.clear();
      si.c.emplace_back(std::pair(0, shape[1]));
      si.h.clear();
      si.h.emplace_back(std::pair(0, shape[2]));
      si.w.clear();
      si.w.emplace_back(std::pair(0, shape[3]));

      tensor_infos[in] = tensor_info_t(si);
      tensor_infos[in].hold_in_lmem = true;

      if (strip_back_judge(in, lg_info, op_set, out_tensor_set)) {
        tensor_branchs.push_back(in);
      }
      continue;
    }
    if (ret == false) {
      return false;
    }
    auto iter = tensor_infos.find(in);
    if (iter != tensor_infos.end()) {
      if (false == is_same_slice_info(si, iter->second.slice_info)) {
        if (module::isCV18xx() || mode == RunMode::TPU_DYNAMIC || !pre_op) {
          return false;
        }
        /* "tpu.store" now not support storing a tensor with margins to GMEM. */
        auto followed_by_store_op = [&lg_info](Operation *op) -> bool {
          if (std::find(lg_info.group_ops.begin(), lg_info.group_ops.end(),
                        op) == lg_info.group_ops.end()) {
            return false; // do not need to a store_op.
          }
          for (auto user : op->getUsers()) {
            if (std::find(lg_info.group_ops.begin(), lg_info.group_ops.end(),
                          user) == lg_info.group_ops.end())
              return true;
          }
          return false;
        };
        // Whether backend func can accept input tensor with margins. (NOTE: LUT
        // has no backend support.)
        auto tpukernel_support_HWmargins = [&lg_info](Operation *op) -> bool {
          if (isa<tpu::Conv2DOp>(op) &&
              op->getAttrOfType<IntegerAttr>("group").getInt() > 1) {
            // Note: backend function is incomplete, feel free to complete it
            // Realizing this functionality could lead to significant rewards.
            return false;
          }
          if (module::getChip() == module::Chip::BM1684) {
            return isa<tpu::Conv2DOp>(op);
          } else {
            return isa<tpu::Conv2DOp, tpu::AddOp, tpu::BinaryShiftOp,
                       tpu::BinaryConstShiftOp>(op);
          }
        };

        if (followed_by_store_op(pre_op)) {
          return false;
        }
        for (auto user : pre_op->getUsers()) {
          if (!(tpukernel_support_HWmargins(user) ||
                isa<tpu::LutOp, tpu::CastOp>(user)) ||
              module::isF8Modes()) {
            return false;
          }
        }

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
          tensor_infos[in] = tensor_info_t(si_both);

          for (auto user : pre_op->getUsers()) {
            if (isa<tpu::LutOp, tpu::CastOp>(user)) {
              // For now, LutOp has no backend support for HW-margins, so
              // HW-margins must insteadly be passed to && processed by its
              // child Operator.
              if (followed_by_store_op(user)) {
                return false;
              }
              for (auto lut_user : user->getUsers()) {
                if (!tpukernel_support_HWmargins(lut_user))
                  return false;
              }
              auto val = user->getResult(0);
              for (int i = 0; i < shape_secs.hsecs; i++) {
                tensor_infos[val].slice_info.h[i].first =
                    tensor_infos[in].slice_info.h[i].first;
                tensor_infos[val].slice_info.h[i].second =
                    tensor_infos[in].slice_info.h[i].second;
              }
              for (int i = 0; i < shape_secs.wsecs; i++) {
                tensor_infos[val].slice_info.w[i].first =
                    tensor_infos[in].slice_info.w[i].first;
                tensor_infos[val].slice_info.w[i].second =
                    tensor_infos[in].slice_info.w[i].second;
              }
            }
          }
          tensor_infos[in].hold_in_lmem = hold_in_lmem;
        } else {
          return false;
        }
      }
    } else {
      tensor_infos[in] = tensor_info_t(si);
      tensor_infos[in].hold_in_lmem = hold_in_lmem;
    }
    if (strip_back_judge(in, lg_info, op_set, out_tensor_set)) {
      tensor_branchs.push_back(in);
    }
  }
  return true;
}

bool stripe_mine_max_slice(const LgInfo &lg_info,
                           const shape_secs_t &shape_secs,
                           TensorInfo &tensor_infos, const LgOptions &options) {
  if (lg_info.group_ops.size() == 1 && false == options.group_by_cores) {
    return true;
  }
  tensor_infos.clear();

  int64_t n, c, d, h, w;
  int64_t max_nslice = 0, max_cslice = 0;
  int64_t max_dslice = 0, max_hslice = 0, max_wslice = 0;
  std::list<Value> tensor_branchs;
  std::multiset<Operation *> op_set;
  ValueSet out_tensor_set;
  slice_info_t si;
  for (auto out : lg_info.group_outs) {
    module::getNCDHW(out, n, c, d, h, w, lg_info.type);
    max_nslice = std::max(max_nslice, ceiling_func(n, shape_secs.nsecs));
    if (Arch::ALIGN_4N) {
      auto stype = module::getStorageType(out);
      int64_t align_n = 32 / stype.getIntOrFloatBitWidth();
      max_nslice = align_up(max_nslice, align_n);
    }
    max_dslice = ceiling_func(d, shape_secs.dsecs); // ?no max?
    max_hslice = ceiling_func(h, shape_secs.hsecs);
    max_wslice = ceiling_func(w, shape_secs.wsecs);
    max_cslice = align_up(ceiling_func(c, shape_secs.csecs), Arch::NPU_NUM);
    si.n.clear();
    si.h.clear();
    si.d.clear();
    si.w.clear();
    si.c.clear();
    si.n.emplace_back(slice_pair_t(0, max_nslice));
    si.h.emplace_back(slice_pair_t(0, max_hslice));
    si.d.emplace_back(slice_pair_t(0, max_dslice));
    si.w.emplace_back(slice_pair_t(0, max_wslice));
    si.c.emplace_back(slice_pair_t(0, max_cslice));
    tensor_infos[out] = tensor_info_t(si);

    out_tensor_set.insert(out);
    if (strip_back_judge(out, lg_info, op_set, out_tensor_set)) {
      tensor_branchs.push_back(out);
    }
  }

  bool ret = false;
  while (!tensor_branchs.empty()) {
    auto out_tensor = tensor_branchs.front();
    tensor_branchs.pop_front();
    ret = backward_update_slice(lg_info, shape_secs, out_tensor, tensor_branchs,
                                tensor_infos, op_set, out_tensor_set);
    if (!ret) {
      return false;
    }
  }

  //  if (check_tensor_slice(tensor_infos) == false) {
  //    return false;
  //  }

  return true;
}

bool stripe_mine_idx_slice(const LgInfo &lg_info,
                           const shape_secs_t &shape_secs,
                           TensorInfo &tensor_infos, const LgOptions &options) {
  if (lg_info.group_ops.size() == 1 && false == options.group_by_cores) {
    return true;
  }
  tensor_infos.clear();

  int64_t n, c, d, h, w;
  std::list<Value> tensor_branchs;
  std::multiset<Operation *> op_set;
  std::set<Value, value_compare> out_tensor_set;
  for (auto out : lg_info.group_outs) {
    module::getNCDHW(out, n, c, d, h, w, lg_info.type);
    auto istype = module::getStorageType(lg_info.group_ins[0]);
    auto ostype = module::getStorageType(out);
    int64_t bitwidth = std::min(istype.getIntOrFloatBitWidth(),
                                ostype.getIntOrFloatBitWidth());
    auto si = get_out_slice_info(shape_secs, n, c, h, d, w, bitwidth);

    tensor_infos[out] = tensor_info_t(si);
    out_tensor_set.insert(out);
    if (strip_back_judge(out, lg_info, op_set, out_tensor_set)) {
      tensor_branchs.push_back(out);
    }
  }

  bool ret = false;
  while (!tensor_branchs.empty()) {
    auto out_tensor = tensor_branchs.front();
    tensor_branchs.pop_front();
    ret = backward_update_slice(lg_info, shape_secs, out_tensor, tensor_branchs,
                                tensor_infos, op_set, out_tensor_set);
    if (!ret) {
      return false;
    }
  }

  return true;
}

void get_max_slice_nchdw(const slice_info_t &slice_info, int64_t &max_nslice,
                         int64_t &max_cslice, int64_t &max_hslice,
                         int64_t &max_dslice, int64_t &max_wslice) {
  max_nslice = 0;
  max_cslice = 0;
  max_hslice = 0;
  max_dslice = 0;
  max_wslice = 0;
  for (auto &slice : slice_info.n) {
    max_nslice = std::max(max_nslice, slice.second);
  }
  for (auto &slice : slice_info.c) {
    max_cslice = std::max(max_cslice, slice.second);
  }
  for (auto &slice : slice_info.h) {
    max_hslice = std::max(max_hslice, slice.second);
  }
  for (auto &slice : slice_info.d) {
    max_dslice = std::max(max_dslice, slice.second);
  }
  for (auto &slice : slice_info.w) {
    max_wslice = std::max(max_wslice, slice.second);
  }
}

std::vector<slice_pair_t>
get_max_slice_nchdw_and_idx(const slice_info_t &slice_info, int64_t &max_nslice,
                            int64_t &max_cslice, int64_t &max_hslice,
                            int64_t &max_dslice, int64_t &max_wslice) {

  std::vector<slice_pair_t> slice_idx;
  max_nslice = 0;
  max_cslice = 0;
  max_hslice = 0;
  max_dslice = 0;
  max_wslice = 0;
  int n_idx = 0;
  int c_idx = 0;
  int h_idx = 0;
  int d_idx = 0;
  int w_idx = 0;
  for (int i = 0; i < slice_info.n.size(); i++) {
    if (slice_info.n[i].second > max_nslice) {
      max_nslice = slice_info.n[i].second;
      n_idx = slice_info.n[i].first;
    }
  }
  slice_idx.push_back({n_idx, max_nslice});

  for (int i = 0; i < slice_info.c.size(); i++) {
    if (slice_info.c[i].second > max_cslice) {
      max_cslice = slice_info.c[i].second;
      c_idx = slice_info.c[i].first;
    }
  }
  slice_idx.push_back({c_idx, max_cslice});

  for (int i = 0; i < slice_info.h.size(); i++) {
    if (slice_info.h[i].second > max_hslice) {
      max_hslice = slice_info.h[i].second;
      h_idx = slice_info.h[i].first;
    }
  }
  slice_idx.push_back({h_idx, max_hslice});

  for (int i = 0; i < slice_info.d.size(); i++) {
    if (slice_info.d[i].second > max_dslice) {
      max_dslice = slice_info.d[i].second;
      d_idx = slice_info.d[i].first;
    }
  }
  slice_idx.push_back({d_idx, max_dslice});

  for (int i = 0; i < slice_info.w.size(); i++) {
    if (slice_info.w[i].second > max_wslice) {
      max_wslice = slice_info.w[i].second;
      w_idx = slice_info.w[i].first;
    }
  }
  slice_idx.push_back({w_idx, max_wslice});

  return slice_idx;
}

int64_t get_buffer_size(Value v, tensor_info_t &ti, group_type_t group_type,
                        Operation *owner_op) {
  int64_t buf_size = 0;
  int64_t n, c, d, h, w;
  module::getNCDHW(v, n, c, d, h, w, group_type);
  bool allow_split = false;
  if (module::isWeight(v)) {
    auto weight_op = dyn_cast<top::WeightOp>(v.getDefiningOp());
    if (weight_op.getAllowSplit() != std::nullopt) {
      allow_split = true;
    }
  }
  if (module::isWeight(v) && allow_split == false) {
    if (group_type == GROUP_SMALL_C) {
      buf_size = Arch::get_tensor_lmem_bytes(v, n, c, d, h, w, ti.eu_align);
    } else {
      buf_size = Arch::get_weight_lmem_bytes(v, group_type, ti.eu_align);
    }
  } else if (module::isDynWeight(v)) { // TODO: need check
    buf_size = Arch::get_weight_lmem_bytes(v, group_type, ti.eu_align);
  } else {
    int64_t nslice, cslice, hslice, dslice, wslice;
    auto si = ti.slice_info;
    if (owner_op) {
      si = ti.slice_infos[owner_op];
    }
    get_max_slice_nchdw(si, nslice, cslice, hslice, dslice, wslice);
    buf_size = Arch::get_tensor_lmem_bytes(v, nslice, cslice, hslice, dslice,
                                           wslice, group_type, ti.eu_align);
  }
  return buf_size;
}

void set_fake_local_layer_param(Operation *op, int64_t nidx, int64_t nslice,
                                int64_t cidx, int64_t cslice, int64_t hidx,
                                int64_t hslice, int64_t didx, int64_t dslice,
                                int64_t widx, int64_t wslice) {
  auto ctx = op->getContext();
  auto builder = OpBuilder(ctx);
  int64_t group_type = 0;
  auto lg_attr = LayerGroupAttr::get(
      ctx, 0, 0, 0, 0, true, false, builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({nidx}),
      builder.getDenseI64ArrayAttr({nslice}),
      builder.getDenseI64ArrayAttr({cidx}),
      builder.getDenseI64ArrayAttr({cslice}),
      builder.getDenseI64ArrayAttr({didx}),
      builder.getDenseI64ArrayAttr({dslice}),
      builder.getDenseI64ArrayAttr({hidx}),
      builder.getDenseI64ArrayAttr({hslice}),
      builder.getDenseI64ArrayAttr({widx}),
      builder.getDenseI64ArrayAttr({wslice}), 0, 0, 0, group_type);
  op->setAttr(LocalGenInterface::kLayerGroupAttrName, lg_attr);
}

// MatMul weight split case
// case1 : [4, 5, 6] * [4, 6, 7] = [4, 5, 7]  => batch = 4, M = 5, k = 6, N = 7
// case2 : [3, 4, 5, 6] * [3, 4, 6, 7] => batch = 12, M = 5, K = 6, N = 7
// case3 :matmul attrs: r_trans=true, hdim_is_batch=true
//=>[1, 16, 320, 64] * [1, 320, 320, 64] => [1, 16, 320, 64] * [1, 64, 320, 320]
//=> batch =320 , M = 16, K = 64, N = 320
// case4: matmul attrs: r_trans=false, l_trans=false, hdim_is_batch=true
//=> 1x248x8x64xf16, 1x64x8x248xf16 -> 1x248x8x248xf16
// other cases TODO
bool check_split_matmul(Operation *op) {
  if (!isa<tpu::MatMulOp>(op)) {
    return false;
  }
  auto matmulOp = dyn_cast<tpu::MatMulOp>(op);

  auto a_s = SmallVector<int64_t>(module::getShape(matmulOp.getInput()));
  auto b_s = SmallVector<int64_t>(module::getShape(matmulOp.getRight()));
  auto o_s = SmallVector<int64_t>(module::getShape(matmulOp.getOutput()));

  if (a_s.size() != b_s.size()) {
    return false;
  }

  // case 1
  if (a_s.size() == 3 && a_s[0] == b_s[0] && a_s[0] != 1 && a_s[2] == b_s[1]) {
    // if(a_s.size() == 3 && /*a_s[0] == b_s[0] && a_s[0] != 1 && */ a_s[2] ==
    // b_s[1]){
    return true;
  }

  // case 2
  if (a_s.size() == 4 && a_s[0] == b_s[0] && a_s[0] != 1 && a_s[1] == b_s[1] &&
      b_s[1] != 1 && a_s[3] == b_s[2]) {
    return true;
  }

  // case 3
  if (matmulOp.getHdimIsBatch() && a_s.size() == 4 && a_s[0] == b_s[0] &&
      a_s[2] == b_s[2] && a_s[3] == b_s[3]) {
    return true;
  }

  // case 4
  if (matmulOp.getHdimIsBatch() && a_s.size() == 4 && a_s[0] == b_s[0] &&
      a_s[2] == b_s[2] && a_s[3] == b_s[1]) {
    return true;
  }
  // other cases
  // TODO

  // Wenet:

  return false;
}

void set_weight_allow_split_attr(Operation *op) {
  auto ctx = op->getContext();
  auto builder = OpBuilder(ctx);
  if (isa<tpu::AddOp, tpu::SubOp, tpu::MulOp, tpu::DivOp, tpu::MinOp,
          tpu::MaxOp, tpu::CompareOp, tpu::ConcatOp, tpu::MatMulOp,
          tpu::BinaryShiftOp>(op) &&
      (module::isWeight(op->getOperand(0)) ||
       module::isWeight(op->getOperand(1)))) {

    if (isa<tpu::MatMulOp>(op) && !check_split_matmul(op)) {
      return;
    }
    top::WeightOp weight_op;
    if (module::isWeight(op->getOperand(0))) {
      weight_op = dyn_cast<top::WeightOp>(op->getOperand(0).getDefiningOp());
    } else if (module::isWeight(op->getOperand(1))) {
      weight_op = dyn_cast<top::WeightOp>(op->getOperand(1).getDefiningOp());
    }
    if (weight_op.getAllowSplit() != std::nullopt) {
      return;
    }
    auto out_shape = module::getShape(weight_op.getResult());
    std::vector<int64_t> AllowSplitVector(out_shape.size(), 1);
    weight_op.setAllowSplitAttr(builder.getI64ArrayAttr(AllowSplitVector));
  } else if (isa<tpu::RopeOp>(op) && (module::isWeight(op->getOperand(1)) ||
                                      module::isWeight(op->getOperand(2)))) {
    if (module::isWeight(op->getOperand(1))) {
      auto weight_op =
          dyn_cast<top::WeightOp>(op->getOperand(1).getDefiningOp());
      if (weight_op.getAllowSplit() == std::nullopt) {
        auto out_shape = module::getShape(weight_op.getResult());
        std::vector<int64_t> AllowSplitVector(out_shape.size(), 1);
        weight_op.setAllowSplitAttr(builder.getI64ArrayAttr(AllowSplitVector));
      }
    }
    if (module::isWeight(op->getOperand(2))) {
      auto weight_op =
          dyn_cast<top::WeightOp>(op->getOperand(2).getDefiningOp());
      if (weight_op.getAllowSplit() == std::nullopt) {
        auto out_shape = module::getShape(weight_op.getResult());
        std::vector<int64_t> AllowSplitVector(out_shape.size(), 1);
        weight_op.setAllowSplitAttr(builder.getI64ArrayAttr(AllowSplitVector));
      }
    }
  }
  return;
}

void delete_weight_allow_split_attr(Operation *op) {
  if (auto weight_op = dyn_cast<top::WeightOp>(op)) {
    if (weight_op.getAllowSplit() == std::nullopt) {
      return;
    }
    op->removeAttr("allow_split");
  }
}

void delete_fake_local_layer_param(Operation *op) {
  op->removeAttr(LocalGenInterface::kLayerGroupAttrName);
}

void generate_fake_global_addr(Operation *op) {
  int64_t offset = Arch::LMEM_BANK_BYTES;
  int64_t i = 0;
  auto ins = get_input_values(op);
  auto outs = get_output_values(op);
  for (auto in : ins) {
    module::setAddress(in, offset * i);
    ++i;
  }
  for (auto out : outs) {
    module::setAddress(out, offset * i);
    ++i;
  }
}

void delete_fake_global_addr(Operation *op) {

  auto ins = get_input_values(op);
  auto outs = get_output_values(op);
  for (auto in : ins) {
    auto type = in.getType().cast<RankedTensorType>();
    Builder builder(in.getContext());
    auto new_type =
        RankedTensorType::get(type.getShape(), type.getElementType());
    in.setType(new_type);
  }
  for (auto out : outs) {
    auto type = out.getType().cast<RankedTensorType>();
    Builder builder(out.getContext());
    auto new_type =
        RankedTensorType::get(type.getShape(), type.getElementType());
    out.setType(new_type);
  }
}

bool is_eu_align_cv18xx(Value opd) {
  auto op = *opd.user_begin();
  if (module::isWeight(opd)) {
    if (isa<tpu::LutOp>(op)) {
      if (opd == op->getOperand(1)) {
        return false;
      }
    } else if (isa<tpu::LutBF16Op>(op)) {
      if (opd == op->getOperand(1) || opd == op->getOperand(2)) {
        return false;
      }
    } else if (isa<tpu::ScaleLutOp>(op)) {
      if (opd == op->getOperand(1)) {
        return false;
      }
    } else if (isa<tpu::ScaleOp>(op)) {
      return false;
    } else if (auto castOp = dyn_cast<tpu::Conv2DOp>(op)) {
      auto attr = castOp.parseParam();
      if (opd == op->getOperand(1) && attr.is_dw) {
        // if is dw conv, filter need to align.
        return true;
      }
      if (module::isUniformQuantized(castOp.getOutput()) &&
          opd == op->getOperand(2) && !attr.is_dw && attr.groups > 1) {
        // if is int8 group conv, bias need to align.
        return true;
      }
      return false;
    } else if (auto castOp = dyn_cast<tpu::DeconvOp>(op)) {
      auto attr = castOp.parseParam();
      if (opd == op->getOperand(1) && attr.is_dw) {
        // if is dw conv, filter need to align.
        return true;
      }
      if (module::isUniformQuantized(castOp.getOutput()) &&
          opd == op->getOperand(2) && !attr.is_dw && attr.g > 1) {
        // if is int8 group conv, bias need to align.
        return true;
      }
      return false;
    } else if (isa<tpu::LayerNormOp>(op)) {
      return false;
    } else {
      return true; // prelu concat
    }
  }
  return true;
}

bool is_eu_align_bm168x(Value opd) {
  auto op = *opd.user_begin();

  if (module::isDynWeight(opd)) {
    return false;
  }

  if (isa<tpu::Conv2DOp, tpu::Conv3DOp, tpu::DeconvOp>(op)) {
    if ((opd == op->getOperand(1) || opd == op->getOperand(2))) {
      return false;
    }
  }

  if (module::isWeight(opd) || module::isTrain()) {
    if (isa<tpu::GroupNormOp, tpu::LayerNormOp, tpu::PixelNormOp,
            tpu::InstanceNormOp>(op)) {
      if ((opd == op->getOperand(1) || opd == op->getOperand(2))) {
        return false;
      }
    } else if (isa<tpu::PReluOp, tpu::ScaleOp>(op)) {
      return false;
    } else if (isa<tpu::BatchNormBwdOp>(op)) {
      if (opd != op->getOperand(0) && opd != op->getOperand(1)) {
        return false;
      }
    } else if (module::isBM1688() || module::isBM1690Family() ||
               module::isSG2380() || module::isMARS3() || module::isSGTPUV8()) {
      if (isa<tpu::RequantIntAxisOp>(op)) {
        if ((opd == op->getOperand(1))) {
          return false;
        }
      }
    }
  }
  return true;
}

bool is_eu_align(Value opd) {
  // Eu align rule may be different in different platforms
  if (module::isCV18xx()) {
    return is_eu_align_cv18xx(opd);
  } else {
    return is_eu_align_bm168x(opd);
  }
}

bool is_value_dont_split(Value opd) {
  if (isa<NoneType>(opd.getType())) {
    return false;
  }

  auto op = *opd.user_begin();
  if (isa<tpu::Conv2DOp, tpu::Conv3DOp, tpu::DeconvOp, tpu::GroupNormOp,
          tpu::LayerNormOp, tpu::PixelNormOp>(op)) {
    if ((opd == op->getOperand(1) || opd == op->getOperand(2))) {
      return true;
    }
  } else if (isa<tpu::PReluOp>(op)) {
    if (opd == op->getOperand(1)) {
      return true;
    }
  } else if (isa<tpu::BatchNormBwdOp>(op)) {
    if (opd != op->getOperand(0) && opd != op->getOperand(1)) {
      return true;
    }
  } else if (isa<tpu::ScaleOp>(op)) {
    if ((opd == op->getOperand(1) || opd == op->getOperand(2))) {
      return true;
    }
  } else if (module::isBM1688() || module::isBM1690Family() ||
             module::isSG2380() || module::isMARS3() || module::isSGTPUV8()) {
    if (isa<tpu::RequantIntAxisOp>(op)) {
      if ((opd == op->getOperand(1))) {
        return true;
      }
    }
  }
  return false;
}

bool need_bcast(Value opd) {
  if (opd.hasOneUse() == false) {
    return false;
  }
  auto use_op = *opd.user_begin();
  if (auto cast_op = dyn_cast<tpu::LutOp>(use_op)) {
    return opd == cast_op.getTable();
  } else if (auto cast_op = dyn_cast<tpu::LutBF16Op>(use_op)) {
    return opd == cast_op.getTable() || opd == cast_op.getMantissa();
  } else if (auto cast_op = dyn_cast<tpu::LRNOp>(use_op)) {
    return opd == cast_op.getTable() || opd == cast_op.getMantissa();
  } else if (auto cast_op = dyn_cast<tpu::LayerNormOp>(use_op)) {
    return module::isCV18xx() && module::isWeight(opd);
  } else {
    return false;
  }
}

int64_t use_3ic(Value opd) {
  for (auto use_op : opd.getUsers()) {
    if (auto cast_op = dyn_cast<tpu::Conv2DOp>(*use_op)) {
      if (opd == cast_op.getInput()) {
        return cast_op.getUse_3icOptimize();
      }
    }
  }
  return 0;
}

int get_user_count_in_group(Value opd, const std::vector<Operation *> &ops) {
  int count = 0;
  for (auto user : opd.getUsers()) {
    if (isa<ReturnOp>(user)) {
      continue;
    }

    if (std::find(ops.begin(), ops.end(), user) != ops.end()) {
      count++;
    }
  }
  return count;
}

std::vector<Value> get_input_values(Operation *op) {
  auto value_vec = std::vector<Value>();
  for (auto in : op->getOperands()) {
    if (in.getType().isa<NoneType>()) {
      continue;
    }
    value_vec.push_back(in);
  }
  return std::move(value_vec);
}

std::vector<Value> get_output_values(Operation *op) {
  auto value_vec = std::vector<Value>();
  for (auto out : op->getResults()) {
    if (out.getType().isa<NoneType>()) {
      continue;
    }
    value_vec.push_back(out);
  }
  return std::move(value_vec);
}

static std::string format_op_in_out_info(Operation *op) {
  std::string tmpStr = " ";
  int64_t n, c, d, h, w;
  // for (auto [index, in] : llvm::enumerate(get_input_values(op))) {
  //   if (is_value_weight(in)) {
  //     module::getNCDHW(in, n, c, d, h, w, GROUP_NORMAL);
  //     tmpStr = tmpStr + llvm::formatv(" in{0}:[{1},{2},{3},{4},{5}]", index,
  //     n,
  //                                     c, d, h, w)
  //                           .str();
  //   }
  // }
  // tmpStr = tmpStr + ", ";
  auto outs = get_output_values(op);
  module::getNCDHW(outs[0], n, c, d, h, w, GROUP_NORMAL);
  tmpStr = tmpStr + llvm::formatv(" out:[{1},{2},{3},{4},{5}], num:{6}", index,
                                  n, c, d, h, w, outs.size())
                        .str();
  return tmpStr;
}

std::shared_ptr<dot_graph> createSubnetGraph(std::vector<Operation *> &ops) {
  std::shared_ptr<dot_graph> dot_graph_log = std::make_shared<dot_graph>();
  std::string pre_op_name, op_name, node_label;
  for (auto op : ops) {
    if (!op)
      continue;
    if (!isa<ReturnOp>(op)) {
      op_name = module::getName(op).str();
      dot_graph_log->add_node_into_graph(op_name);
      dot_graph_log->add_node_label(op_name,
                                    op->getName().getStringRef().str() +
                                        format_op_in_out_info(op));
      bool next_layer_has_return = false;
      for (auto itr = op->user_begin(); itr != op->user_end(); itr++) {
        if (!isa<ReturnOp>(*itr)) {
          auto to = module::getName(*itr).str();
          dot_graph_log->add_node_into_graph(to);
          dot_graph_log->add_node_label(to,
                                        (*itr)->getName().getStringRef().str() +
                                            format_op_in_out_info(*itr));
          dot_graph_log->add_edge_into_graph(op_name, to);
        } else {
          next_layer_has_return = true;
        }
      }
      if (next_layer_has_return) {
        dot_graph_log->add_node_label(op_name, std::string("to_returnOp"));
      }

      for (auto v : op->getOperands()) {
        auto pre_op = v.getDefiningOp();
        if (pre_op && isa<top::NoneOp>(pre_op)) {
          continue;
        }
        if (std::find(ops.begin(), ops.end(), pre_op) == ops.end()) {
          if (pre_op) {
            pre_op_name = module::getName(pre_op).str();
            node_label = pre_op->getName().getStringRef().str() +
                         format_op_in_out_info(pre_op);
          } else {
            pre_op_name =
                "arg" + std::to_string(v.cast<BlockArgument>().getArgNumber());
            auto shape = v.getType().cast<RankedTensorType>().getShape();
            node_label = "[";
            for (auto i : shape) {
              node_label = node_label + std::to_string(i) + ",";
            }
            node_label.back() = ']';
          }
          dot_graph_log->add_node_into_graph(pre_op_name);
          dot_graph_log->add_node_label(pre_op_name, node_label);
          dot_graph_log->add_edge_into_graph(pre_op_name, op_name);
        }
      }
    }
  }
  return dot_graph_log;
}

} // namespace tpu
} // namespace tpu_mlir

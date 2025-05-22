//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <chrono>
#include <fstream>

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/InternalOptimizer.h"
#include "tpu_mlir/Support/TopoSorter.h"

using namespace tpu_mlir::tpu;
using namespace tpu_mlir::backend;

bool pair_int_Sort_by_int(const std::pair<int, int> &v1,
                          const std::pair<int, int> &v2) {
  return v1.second < v2.second;
}

void DfsFindNextMatMul(Operation *start_op, Operation *&next_matmul_op) {
  for (auto user : start_op->getUsers()) {
    if (isa<ReturnOp>(user)) {
      continue;
    }
    // if (auto lg_op = dyn_cast<LocalGenInterface>(user)) { }
    if (isa<tpu::MatMulOp>(user)) {
      next_matmul_op = user;
      return;
    } else if (module::isInMatMulGrpOp(user)) {
      DfsFindNextMatMul(user, next_matmul_op);
    }
  }
}

void BfsFindNextMatMul(Operation *start_op, Operation *&next_matmul_op,
                       std::vector<Operation *> &subnet_ops) {
  std::queue<Operation *> q;
  for (auto user : start_op->getUsers()) {
    if (!isa<ReturnOp>(user)) {
      if (std::find(subnet_ops.begin(), subnet_ops.end(), user) !=
          subnet_ops.end()) {
        q.push(user);
      }
    }
  }

  while (!q.empty()) {
    auto op = q.front();
    q.pop();
    if (isa<tpu::MatMulOp>(op)) {
      next_matmul_op = op;
      return;
    }
    if (isa<tpu::SoftmaxOp>(op) || module::isInMatMulGrpOp(op)) {
      for (auto user : op->getUsers()) {
        if (!isa<ReturnOp>(user)) {
          if (std::find(subnet_ops.begin(), subnet_ops.end(), user) !=
              subnet_ops.end()) {
            q.push(user);
          }
        }
      }
    }
  }
}

void FindValidOpUntilMatMul(Operation *start_op,
                            std::vector<Operation *> &find_ops,
                            std::vector<Operation *> &next_matmul_pre_ops,
                            bool &find_softmax, bool &failed) {
  for (auto user : start_op->getUsers()) {
    if (isa<ReturnOp>(user)) {
      continue;
    }
    if (std::find(next_matmul_pre_ops.begin(), next_matmul_pre_ops.end(),
                  user) == next_matmul_pre_ops.end()) {
      continue;
    }
    if (isa<tpu::MatMulOp>(user)) {
      if (std::find(find_ops.begin(), find_ops.end(), user) == find_ops.end()) {
        find_ops.push_back(user);
      }
      return;
    }

    if (isa<tpu::SoftmaxOp>(user)) {
      llvm::errs() << "find_softmax op:" << module::getName(user).str() << '\n';
      find_softmax = true;
    }
    if (isa<tpu::SoftmaxOp>(user) || module::isInMatMulGrpOp(user)) {
      if (std::find(find_ops.begin(), find_ops.end(), user) == find_ops.end()) {
        llvm::errs() << "find op:" << module::getName(user).str() << "\n";
        find_ops.push_back(user);
        FindValidOpUntilMatMul(user, find_ops, next_matmul_pre_ops,
                               find_softmax, failed);
      }
    } else {
      llvm::errs() << "fail2 at op:" << module::getName(user).str() << "\n";
      failed = true;
      return;
    }
  }
}

bool speical_layer_group_base::search_two_mmOp(
    Operation *start_op, Operation *&next_mmOp,
    std::vector<Operation *> &subnet_ops) {
  if (isa<tpu::MatMulOp>(start_op)) {
    llvm::errs() << "start_op:" << module::getName(start_op).str() << '\n';
    next_mmOp = nullptr;
    BfsFindNextMatMul(start_op, next_mmOp, subnet_ops);
    if (next_mmOp) {
      main_mm_op = name() == "mlp_group" ? start_op : next_mmOp;
      llvm::errs() << "next_mmOp:" << module::getName(next_mmOp).str() << '\n';
      std::vector<Operation *> pre_ops;
      find_all_pre_ops(next_mmOp, pre_ops, &subnet_ops);
      ops.clear();
      bool failed = false;
      FindValidOpUntilMatMul(start_op, ops, pre_ops, find_softmax, failed);
      if (!failed) {
        ops.push_back(start_op);
        auto ops_reorder = sortOpsByOtherOpsOrder(subnet_ops, ops);
        ops.assign(ops_reorder.begin(), ops_reorder.end());
        return true;
      }
    }
  }
  return false;
}

void speical_layer_group_base::get_batch_size(shape_secs_t &shape_secs) {
  int64_t in_n, in_c, in_d, in_h, in_w;
  Operation *op = main_mm_op;
  llvm::errs() << "main_mm_op:" << op << '\n';
  llvm::errs() << "main_mm_op:" << show_op_info(op) << '\n';
  module::getNCDHW(op->getOperand(0), in_n, in_c, in_d, in_h, in_w,
                   GROUP_MM_OPT3);
  shape_secs.n = in_n;
  shape_secs.c = in_c;
  auto mm_op = dyn_cast<tpu::MatMulOp>(op);
  if (mm_op.getHdimIsBatch()) {
    auto shape =
        op->getOperand(0).getType().cast<RankedTensorType>().getShape();
    assert(shape.size() == 4);
    for (int i = 1; i <= shape[0]; i++) {
      map_n_slice_num_to_max_n[i] = shape[2] * ceiling_func(shape[0], i);
    }
    for (int i = 2; i <= shape[2]; i++) {
      map_n_slice_num_to_max_n[shape[0] * i] = ceiling_func(shape[2], i);
    }
  }

  if (mm_op.getLeftTranspose()) {
    shape_secs.c = in_h;
  }
  module::getNCDHW(op->getOperand(1), in_n, in_c, in_d, in_h, in_w,
                   GROUP_MM_OPT3);
  shape_secs.h = in_h;

  if (mm_op.getRightTranspose()) {
    shape_secs.h = in_c;
  }
  if (!col_cut ||
      (name() == "mlp_group" && module::getChip() != module::Chip::BM1690)) {
    shape_secs.h = 1;
  }
  llvm::errs() << "get matmul group n:" << shape_secs.n
               << ", c:" << shape_secs.c << ", h:" << shape_secs.h << '\n';
}

bool speical_layer_group_base::update_shape_secs_for_ilp_group(
    shape_secs_t &shape_secs, const shape_secs_t &max_shape_secs) {
  bool updated = false;
  if (shape_secs.n_slice_num == shape_secs.n) {
    if (shape_secs.c_slice_num == shape_secs.c) {
      if (shape_secs.h_slice_num < shape_secs.h) {
        shape_secs.h_slice_num++;
        updated = true;
      }
    } else {
      shape_secs.c_slice_num++;
      updated = true;
    }
  } else {
    if (shape_secs.n_slice_num < shape_secs.n) {
      shape_secs.n_slice_num++;
      updated = true;
    }
  }
  return updated;
}

void speical_layer_group_base::fill_slice_info(ilp_LgInfo &ilp_lg_info) {
  int64_t n, c, d, h, w;
  ilp_lg_info.tensor_infos.clear();
  ilp_lg_info.value_store_to_l2m.clear();
  ilp_lg_info.value_load_to_l2m.clear();
  int64_t n_slice_num = ilp_lg_info.shape_secs.n_slice_num;
  llvm::errs() << "n_slice_num: " << n_slice_num
               << ", c_slice_num: " << ilp_lg_info.shape_secs.c_slice_num
               << ", h_slice_num: " << ilp_lg_info.shape_secs.h_slice_num
               << "\n";

  if (hdim_is_batch) {
    assert(name() == "attention_group");
    int64_t n_slice, h_slice;
    auto shape = main_mm_op->getOperand(0)
                     .getType()
                     .cast<RankedTensorType>()
                     .getShape()
                     .vec();
    if (n_slice_num > shape[0]) {
      n_slice = shape[0];
      h_slice = n_slice_num / shape[0];
    } else {
      n_slice = n_slice_num;
      h_slice = 1;
    }
    ilp_lg_info.shape_secs.shape_0 = shape[0];

    for (auto op : ilp_lg_info._lgInfo.group_ops) {
      for (auto in : get_input_values(op)) {
        shape = in.getType().cast<RankedTensorType>().getShape().vec();
        slice_info_t si;
        slice_distributor(si.n, shape[0], n_slice);
        slice_distributor(si.h, shape[2], h_slice);
        slice_distributor(si.d, 1, 1);
        if (module::IsRightMat(in)) {
          ilp_lg_info.value_load_to_l2m[in] = -1;
          slice_distributor(si.c, shape[1], 1);
          if (ilp_lg_info._lgInfo.group_ops.back() == op) {
            slice_distributor(si.w, shape[3],
                              ilp_lg_info.shape_secs.h_slice_num);
          } else {
            slice_distributor(si.w, shape[3], 1);
          }
        } else {
          slice_distributor(si.c, shape[1], ilp_lg_info.shape_secs.c_slice_num);
          slice_distributor(si.w, shape[3], 1);
        }
        tensor_info_t t_info(si);
        t_info.eu_align = true;
        ilp_lg_info.tensor_infos[in] = t_info;
      }

      for (auto out : get_output_values(op)) {
        shape = out.getType().cast<RankedTensorType>().getShape().vec();
        slice_info_t si;
        slice_distributor(si.n, shape[0], n_slice);
        slice_distributor(si.c, shape[1], ilp_lg_info.shape_secs.c_slice_num);
        slice_distributor(si.d, 1, 1);
        slice_distributor(si.h, shape[2], h_slice);
        if (ilp_lg_info._lgInfo.group_ops.back() == op) {
          slice_distributor(si.w, shape[3], ilp_lg_info.shape_secs.h_slice_num);
        } else {
          slice_distributor(si.w, shape[3], 1);
        }
        tensor_info_t t_info(si);
        t_info.eu_align = true;
        ilp_lg_info.tensor_infos[out] = t_info;
      }
    }
    return;
  }

  for (auto op : ilp_lg_info._lgInfo.group_ops) {
    for (auto in : get_input_values(op)) {
      module::getNCDHW(in, n, c, d, h, w, ilp_lg_info._lgInfo.type);
      slice_info_t si;
      slice_distributor(si.n, n, n_slice_num);
      slice_distributor(si.c, c, 1);
      slice_distributor(si.d, d, 1);
      slice_distributor(si.h, h, 1);
      slice_distributor(si.w, w, 1);
      if (module::IsRightMat(in)) {
        if (name() == "mlp_group") {
          if (ilp_lg_info._lgInfo.group_ops[0] == op) {
            llvm::errs() << "in: " << module::getName(in).str()
                         << ", cut h to h_slice_num\n";
            slice_distributor(si.h, h, ilp_lg_info.shape_secs.h_slice_num);
          } else {
            llvm::errs() << "in: " << module::getName(in).str()
                         << ", cut c to h_slice_num\n";
            slice_distributor(si.c, c, ilp_lg_info.shape_secs.h_slice_num);
          }
        } else if (name() == "attention_group") {
          if (ilp_lg_info._lgInfo.group_ops.back() == op) {
            llvm::errs() << "in: " << module::getName(in).str()
                         << ", cut h to h_slice_num\n";
            slice_distributor(si.h, h, ilp_lg_info.shape_secs.h_slice_num);
          }
        } else if (name() == "single_matmul_group") {
          llvm::errs() << "in: " << module::getName(in).str()
                       << ", cut h to h_slice_num\n";
          slice_distributor(si.h, h, ilp_lg_info.shape_secs.h_slice_num);
        }
        ilp_lg_info.value_load_to_l2m[in] = -1;
      } else {
        slice_distributor(si.c, c, ilp_lg_info.shape_secs.c_slice_num);
        llvm::errs() << "in: " << module::getName(in).str()
                     << ", cut c to c_slice_num\n";
        if (name() == "mlp_group") {
          if (isa<tpu::MatMulOp>(op)) {
            if (ilp_lg_info._lgInfo.group_ops[0] != op) {
              slice_distributor(si.h, h, ilp_lg_info.shape_secs.h_slice_num);
              llvm::errs() << "in: " << module::getName(in).str()
                           << ", cut h to h_slice_num\n";
            }
          } else {
            slice_distributor(si.h, h, ilp_lg_info.shape_secs.h_slice_num);
          }
        } else if (name() == "single_matmul_group") {
          if (std::find(h_cut_ops.begin(), h_cut_ops.end(), op) !=
              h_cut_ops.end()) {
            llvm::errs() << "in: " << module::getName(in).str()
                         << ", cut h to h_slice_num\n";
            slice_distributor(si.h, h, ilp_lg_info.shape_secs.h_slice_num);
          }
        }
      }
      tensor_info_t t_info(si);
      t_info.eu_align = true;
      ilp_lg_info.tensor_infos[in] = t_info;
    }

    for (auto out : get_output_values(op)) {
      module::getNCDHW(out, n, c, d, h, w, ilp_lg_info._lgInfo.type);
      llvm::errs() << "out: " << module::getName(out).str()
                   << ", cut n/c to n/c_slice_num\n";
      slice_info_t si;
      slice_distributor(si.n, n, n_slice_num);
      slice_distributor(si.c, c, ilp_lg_info.shape_secs.c_slice_num);
      slice_distributor(si.d, d, 1);
      slice_distributor(si.h, h, 1);
      slice_distributor(si.w, w, 1);
      if (name() == "mlp_group") {
        if (ilp_lg_info._lgInfo.group_ops.back() != op) {
          llvm::errs() << "out: " << module::getName(out).str()
                       << ", cut h to h_slice_num\n";
          slice_distributor(si.h, h, ilp_lg_info.shape_secs.h_slice_num);
        } else {
          if (ilp_lg_info.shape_secs.h_slice_num > 1) {
            ilp_lg_info.value_store_to_l2m[out] = -1;
          }
        }
      } else if (name() == "attention_group") {
        if (ilp_lg_info._lgInfo.group_ops.back() == op) {
          llvm::errs() << "out: " << module::getName(out).str()
                       << ", cut h to h_slice_num\n";
          slice_distributor(si.h, h, ilp_lg_info.shape_secs.h_slice_num);
        }
      } else if (name() == "single_matmul_group") {
        if (isa<tpu::MatMulOp>(op) ||
            std::find(h_cut_ops.begin(), h_cut_ops.end(), op) !=
                h_cut_ops.end()) {
          llvm::errs() << "out: " << module::getName(out).str()
                       << ", cut h to h_slice_num\n";
          slice_distributor(si.h, h, ilp_lg_info.shape_secs.h_slice_num);
        }
      }
      tensor_info_t t_info(si);
      t_info.eu_align = true;
      ilp_lg_info.tensor_infos[out] = t_info;
    }
  }
}

bool speical_layer_group_base::inc_n_slice_num(int &n_slice_num,
                                               int max_n_slice_num) {
  if (map_n_slice_num_to_max_n.size() > 0) {
    if (n_slice_num < map_n_slice_num_to_max_n.rbegin()->first) {
      for (auto itr : map_n_slice_num_to_max_n) {
        if (itr.first > n_slice_num) {
          n_slice_num = itr.first;
          llvm::errs() << "get new n_slice_num:" << n_slice_num << "\n";
          return true;
        }
      }
    }
  } else {
    if (n_slice_num < max_n_slice_num) {
      n_slice_num++;
      llvm::errs() << "inc n_slice_num, get:" << n_slice_num << "\n";
      return true;
    }
  }
  return false;
}

bool speical_layer_group_base::inc_slice_num(
    Operation *op, int &n_slice_num, int &try_c_slice_num, int &try_h_slice_num,
    int max_n_slice_num, int max_c_slice_num, int max_h_slice_num,
    int old_target_secs, bool inc_c_slice) {
  bool tmp_inc_c_slice = inc_c_slice;
  int old_n_slice_num = n_slice_num, old_try_c_slice_num = try_c_slice_num,
      old_try_h_slice_num = try_h_slice_num;
  for (int i = 0; i < 3;
       i++) { // why 3? If the first two fail, do the first one again
    if (!inc_n_slice_num(n_slice_num, max_n_slice_num)) {
      if (tmp_inc_c_slice) {
        if (try_c_slice_num < max_c_slice_num) {
          try_c_slice_num++;
          llvm::errs() << "inc_c_slice, try_c_slice_num++ ->" << try_c_slice_num
                       << "\n";
        } else {
          if (try_h_slice_num < max_h_slice_num && is_cut_h(op)) {
            try_h_slice_num++;
            llvm::errs() << "inc_c_slice, try_h_slice_num++ ->"
                         << try_h_slice_num << "\n";
          } else {
            llvm::errs() << "inc_c_slice, inc_slice_num fail1\n";
            return false;
          }
        }
      } else {
        if (try_h_slice_num < max_h_slice_num && is_cut_h(op)) {
          try_h_slice_num++;
          llvm::errs() << "try_h_slice_num++ ->" << try_h_slice_num << "\n";
        } else {
          if (try_c_slice_num < max_c_slice_num) {
            try_c_slice_num++;
            llvm::errs() << "try_c_slice_num++ ->" << try_c_slice_num << "\n";
          } else {
            llvm::errs() << "inc_slice_num fail2\n";
            return false;
          }
        }
      }
    }

    int secs = get_secs(op, n_slice_num, try_c_slice_num, try_h_slice_num);
    if (secs <= old_target_secs) {
      llvm::errs() << "get final secs:" << secs << "\n";
      break;
    }
    tmp_inc_c_slice = tmp_inc_c_slice ? false : true;
    if (i < 2) {
      n_slice_num = old_n_slice_num;
      try_c_slice_num = old_try_c_slice_num;
      try_h_slice_num = old_try_h_slice_num;
    }
  }
  return true;
}

bool speical_layer_group_base::is_cut_h(Operation *op) {
  if (name() == "single_matmul_group" || name() == "mlp_group" ||
      (name() == "attention_group" && isa<tpu::MatMulOp>(op) &&
       op == ops.back())) {
    return true;
  }
  return false;
}

bool BfsCheckOutValue(Value value, std::vector<Operation *> &grp_ops) {
  std::vector<Operation *> checked_ops;
  std::queue<Operation *> q;
  for (auto user : value.getUsers()) {
    if (!isa<ReturnOp>(user)) {
      if (std::find(grp_ops.begin(), grp_ops.end(), user) == grp_ops.end()) {
        q.push(user);
      }
    }
  }

  while (!q.empty()) {
    auto op = q.front();
    checked_ops.push_back(op);
    q.pop();
    if (std::find(grp_ops.begin(), grp_ops.end(), op) != grp_ops.end()) {
      llvm::errs() << "BfsCheckOutValue fail\n";
      return false;
    }

    for (auto user : op->getUsers()) {
      if (!isa<ReturnOp>(user)) {
        if (std::find(grp_ops.begin(), grp_ops.end(), user) == grp_ops.end()) {
          if (std::find(checked_ops.begin(), checked_ops.end(), user) !=
              checked_ops.end()) {
            q.push(user);
          }
        } else {
          llvm::errs() << "BfsCheckOutValue fail\n";
          return false;
        }
      }
    }
  }
  return true;
}

bool speical_layer_group_base::check_group_valid() {
  LgInfo lgInfo;
  lgInfo.group_ops.assign(ops.begin(), ops.end());
  lgInfo.update_group_io();
  bool valid = true;
  for (auto out : lgInfo.group_outs) {
    if (!BfsCheckOutValue(out, ops)) {
      valid = false;
      break;
    }
  }

  return valid;
}

int speical_layer_group_base::get_secs(Operation *op, int n_slice_num,
                                       int c_slice_num, int h_slice_num) {
  int secs = 0;
  if (is_cut_h(op)) {
    secs = h_slice_num * c_slice_num * n_slice_num;
  } else {
    secs = c_slice_num * n_slice_num;
  }
  return secs;
}

int speical_layer_group_base::get_slice_max_n(int n, int slice_num) {
  if (map_n_slice_num_to_max_n.size() > 0) {
    int pre_num = 0;
    for (auto itr : map_n_slice_num_to_max_n) {
      if (slice_num < itr.first) {
        return pre_num;
      }
      pre_num = itr.second;
    }
    return pre_num;
  }
  return ceiling_func(n, slice_num);
}

int speical_layer_group_base::get_best_n_slice_num(int n,
                                                   int expect_slice_num) {
  if (map_n_slice_num_to_max_n.size() > 0) {
    int pre_num = 0;
    for (auto itr : map_n_slice_num_to_max_n) {
      if (expect_slice_num < itr.first) {
        return pre_num;
      }
      pre_num = itr.first;
    }
    return pre_num;
  } else {
    if (n >= expect_slice_num) {
      return expect_slice_num;
    } else {
      return n;
    }
  }
}

bool speical_layer_group_base::CalcMatMulGroupTpNum(ilp_LgInfo &lg_info,
                                                    Operation *&failed_op,
                                                    int64_t core_num) {
  int64_t in_n, in_c, in_d, in_h, in_w, out_n, out_c, out_d, out_h, out_w;
  group_type_t type = lg_info._lgInfo.type;
  lg_info.p_special_grp->get_batch_size(lg_info.shape_secs);
  int batch_size = lg_info.shape_secs.n;
  int glo_n_slice_num = get_best_n_slice_num(batch_size, core_num),
      glo_c_slice_num = 1, glo_h_slice_num = 1;

  std::vector<Operation *> tmp_ops, tmp_ops2;
  for (auto op : lg_info._lgInfo.group_ops) {
    if (isa<tpu::MatMulOp>(op)) {
      tmp_ops.push_back(op);
    } else {
      tmp_ops2.push_back(op);
    }
  }
  if (name() == "attention_group" && tmp_ops.size() == 2) {
    std::swap(tmp_ops[0], tmp_ops[1]);
  }
  tmp_ops.insert(tmp_ops.end(), tmp_ops2.begin(), tmp_ops2.end());
  bool first_matmul = true;
  for (auto op : tmp_ops) {
    auto ins = get_input_values(op);
    auto outs = get_output_values(op);
    int try_n_slice_num = glo_n_slice_num, try_c_slice_num = glo_c_slice_num,
        try_h_slice_num = glo_h_slice_num;
    int pre_valid_loc_slice_h = glo_h_slice_num,
        pre_valid_loc_slice_c = glo_c_slice_num;
    int slice_max_n = get_slice_max_n(batch_size, glo_n_slice_num),
        pre_valid_loc_slice_n = glo_n_slice_num;
    llvm::errs() << "CalcMatMulGroupTpNum for op:" << module::getName(op).str()
                 << '\n';
    llvm::errs() << "slice_max_n:" << slice_max_n << '\n';
    int secs = get_secs(op, glo_n_slice_num, try_c_slice_num, try_h_slice_num);
    int old_target_secs = align(secs, core_num);
    bool init_secs_is_ok = old_target_secs == secs;
    do {
      module::getNCDHW(ins[0], in_n, in_c, in_d, in_h, in_w, type);
      llvm::errs() << "in0_n:" << in_n << ", in_c:" << in_c << ", in_h:" << in_h
                   << ", in_w:" << in_w << '\n';
      if (in_n != batch_size && module::IsReshapeOpInOrOut(ins[0])) {
        failed_op = op;
        llvm::errs() << "in_n != batch_size at op:" << module::getName(op).str()
                     << '\n';
        return false;
      }
      int64_t in0_lmem_bytes = 0, in1_lmem_bytes = 0, in2_lmem_bytes = 0,
              out0_lmem_bytes = 0;
      in_c = align(in_c, try_c_slice_num) / try_c_slice_num;
      if (name() == "mlp_group") {
        if (!isa<tpu::MatMulOp>(op) || op != lg_info._lgInfo.group_ops[0]) {
          in_h = align(in_h, try_h_slice_num) / try_h_slice_num;
        }
      } else if (name() == "single_matmul_group") {
        if (std::find(h_cut_ops.begin(), h_cut_ops.end(), op) !=
            h_cut_ops.end()) {
          in_h = align(in_h, try_h_slice_num) / try_h_slice_num;
        }
      }
      llvm::errs() << "new in_c:" << in_c << ", in_h:" << in_h << '\n';
      in0_lmem_bytes = align_64(Arch::get_tensor_lmem_bytes(
          ins[0], slice_max_n, in_c, in_d, in_h, in_w));

      module::getNCDHW(outs[0], out_n, out_c, out_d, out_h, out_w, type);
      llvm::errs() << "out0_n:" << out_n << ", out_c:" << out_c
                   << ", out_h:" << out_h << ", out_w:" << out_w << '\n';
      if (out_n != batch_size && module::IsReshapeOpInOrOut(outs[0])) {
        failed_op = op;
        llvm::errs() << "out_n != batch_size at op:"
                     << module::getName(op).str() << '\n';
        return false;
      }
      out_c = align(out_c, try_c_slice_num) / try_c_slice_num;
      if (name() == "mlp_group") {
        if (!isa<tpu::MatMulOp>(op) || op == lg_info._lgInfo.group_ops[0]) {
          out_h = align(out_h, try_h_slice_num) / try_h_slice_num;
        }
      } else if (name() == "attention_group") {
        if (isa<tpu::MatMulOp>(op) && op == ops.back()) {
          out_h = align(out_h, try_h_slice_num) / try_h_slice_num;
        }
      } else if (name() == "single_matmul_group") {
        if (isa<tpu::MatMulOp>(op) ||
            std::find(h_cut_ops.begin(), h_cut_ops.end(), op) !=
                h_cut_ops.end()) {
          out_h = align(out_h, try_h_slice_num) / try_h_slice_num;
        }
      }
      llvm::errs() << "new out_c:" << out_c << ", out_h:" << out_h << '\n';
      out0_lmem_bytes = align_64(Arch::get_tensor_lmem_bytes(
          outs[0], slice_max_n, out_c, out_d, out_h, out_w));
      if (name() == "attention_group") {
        if (isa<tpu::MatMulOp>(op) && try_h_slice_num > 1 && op == ops.back()) {
          // The second matmul input and output occupy twice the memory to
          // consider the store of the previous time slot and the load of the
          // next time slot
          out0_lmem_bytes *= 2;
        }
      }

      auto lg_op = cast<LocalGenInterface>(op);
      int64_t buffer_size = lg_op.getBufferSize(
          in0_lmem_bytes, out0_lmem_bytes, slice_max_n, in_c, in_h, in_d, in_w,
          slice_max_n, out_c, out_h, out_d, out_w, type);
      if (ins.size() > 1) {
        module::getNCDHW(ins[1], in_n, in_c, in_d, in_h, in_w, type);
        llvm::errs() << "in1_n:" << in_n << ", in_c:" << in_c
                     << ", in_h:" << in_h << ", in_w:" << in_w << '\n';
        if (in_n != batch_size && module::IsReshapeOpInOrOut(ins[1])) {
          failed_op = op;
          llvm::errs() << "for ins[1], in_n != batch_size at op:"
                       << module::getName(op).str() << '\n';
          return false;
        }
        if (name() == "mlp_group") {
          if (isa<tpu::MatMulOp>(op)) {
            if (op == lg_info._lgInfo.group_ops[0]) {
              if (dyn_cast<tpu::MatMulOp>(op).getRightTranspose()) {
                in_c = align(in_c, try_h_slice_num) / try_h_slice_num;
              } else {
                in_h = align(in_h, try_h_slice_num) / try_h_slice_num;
              }
            } else {
              if (dyn_cast<tpu::MatMulOp>(op).getRightTranspose()) {
                in_h = align(in_h, try_h_slice_num) / try_h_slice_num;
              } else {
                // Note that the number of rows for the right matrix of the
                // second matml is try_h_slice_num
                in_c = align(in_c, try_h_slice_num) / try_h_slice_num;
              }
            }
          } else {
            in_c = align(in_c, try_c_slice_num) / try_c_slice_num;
            in_h = align(in_h, try_h_slice_num) / try_h_slice_num;
          }
        } else if (name() == "attention_group") {
          if (isa<tpu::MatMulOp>(op) && op == ops.back()) {
            in_h = align(in_h, try_h_slice_num) / try_h_slice_num;
          }
        } else if (name() == "single_matmul_group") {
          in_c = align(in_c, try_c_slice_num) / try_c_slice_num;
          if (isa<tpu::MatMulOp>(op) ||
              std::find(h_cut_ops.begin(), h_cut_ops.end(), op) !=
                  h_cut_ops.end()) {
            in_h = align(in_h, try_h_slice_num) / try_h_slice_num;
          }
        }
        llvm::errs() << "new in1_c:" << in_c << ", in1_h:" << in_h << '\n';
        if (in_n != batch_size && in_n == 1) {
          in1_lmem_bytes = align_64(
              Arch::get_tensor_lmem_bytes(ins[1], 1, in_c, in_d, in_h, in_w));
        } else {
          in1_lmem_bytes = align_64(Arch::get_tensor_lmem_bytes(
              ins[1], slice_max_n, in_c, in_d, in_h, in_w));
        }
        if (name() == "attention_group") {
          if (isa<tpu::MatMulOp>(op)) {
            if (op == ops.back()) {
              if (try_h_slice_num > 1) {
                in1_lmem_bytes *= 2;
              }
            } else {
              if (in1_lmem_bytes >= Arch::LMEM_BYTES) {
                llvm::errs() << "the right matrix of Matmul before softmax is "
                                "too large, op:"
                             << module::getName(op).str() << "\n";
                failed_op = op;
                return false;
              }
            }
          }
        }
      }

      if (ins.size() > 2) {
        assert(isa<tpu::MatMulOp>(op));
        module::getNCDHW(ins[2], in_n, in_c, in_d, in_h, in_w, type);
        llvm::errs() << "in2_n:" << in_n << ", in_c:" << in_c
                     << ", in_h:" << in_h << ", in_w:" << in_w << '\n';
        in_h = align(in_h, try_h_slice_num) / try_h_slice_num;
        llvm::errs() << "new in2_h:" << in_h << '\n';
        if (in_n != batch_size && in_n == 1) {
          in2_lmem_bytes = align_64(
              Arch::get_tensor_lmem_bytes(ins[2], 1, in_c, in_d, in_h, in_w));
        } else {
          in2_lmem_bytes = align_64(Arch::get_tensor_lmem_bytes(
              ins[2], slice_max_n, in_c, in_d, in_h, in_w));
        }
        if (name() == "attention_group") {
          if (op == ops.back()) {
            if (try_h_slice_num > 1) {
              in2_lmem_bytes *= 2;
            }
          } else {
            if (in2_lmem_bytes >= Arch::LMEM_BYTES) {
              llvm::errs() << "the bias matrix of Matmul before softmax is too "
                              "large, op:"
                           << module::getName(op).str() << "\n";
              failed_op = op;
              return false;
            }
          }
        }
      }

      bool inc_c_slice = true;
      if (isa<tpu::MatMulOp>(op)) {
        if (name() != "attention_group" || op == ops.back()) {
          inc_c_slice = in0_lmem_bytes >= in1_lmem_bytes; //相等时优先切c
        }
      }

      int total =
          buffer_size + in0_lmem_bytes + in1_lmem_bytes + out0_lmem_bytes;
      bool mem_enough = total <= Arch::LMEM_BYTES;
      llvm::errs() << "in0_lmem_bytes:" << in0_lmem_bytes
                   << ", out0_lmem_bytes:" << out0_lmem_bytes
                   << ", in1_lmem_bytes:" << in1_lmem_bytes
                   << ", buffer_size:" << buffer_size << ", total:" << total
                   << ", inc_c_slice:" << inc_c_slice
                   << ", old_target_secs:" << old_target_secs
                   << ", mem_enough:" << mem_enough << '\n';
      if (mem_enough) {
        if (init_secs_is_ok || !first_matmul) {
          llvm::errs() << "init_secs_is_ok\n";
          break;
        }
        if (!inc_slice_num(op, try_n_slice_num, try_c_slice_num,
                           try_h_slice_num, batch_size, lg_info.shape_secs.c,
                           lg_info.shape_secs.h, old_target_secs,
                           inc_c_slice)) {
          failed_op = op;
          return false;
        }
        secs = get_secs(op, try_n_slice_num, try_c_slice_num, try_h_slice_num);
        if (secs > old_target_secs) {
          llvm::errs() << "new secs(" << secs
                       << ") >= old_target_secs, break\n";
          break;
        }
      } else {
        if (!inc_slice_num(op, try_n_slice_num, try_c_slice_num,
                           try_h_slice_num, batch_size, lg_info.shape_secs.c,
                           lg_info.shape_secs.h, old_target_secs,
                           inc_c_slice)) {
          failed_op = op;
          return false;
        }
        secs = get_secs(op, try_n_slice_num, try_c_slice_num, try_h_slice_num);
        old_target_secs = align(secs, core_num);
        init_secs_is_ok = old_target_secs == secs;
      }
      slice_max_n = get_slice_max_n(batch_size, try_n_slice_num);
      llvm::errs() << "update slice_max_n:" << slice_max_n << '\n';
      pre_valid_loc_slice_n = try_n_slice_num;
      pre_valid_loc_slice_c = try_c_slice_num;
      pre_valid_loc_slice_h = try_h_slice_num;
    } while (true);

    if (pre_valid_loc_slice_c > glo_c_slice_num) {
      glo_c_slice_num = pre_valid_loc_slice_c;
    }
    if (pre_valid_loc_slice_h > glo_h_slice_num) {
      glo_h_slice_num = pre_valid_loc_slice_h;
    }
    if (pre_valid_loc_slice_n > glo_n_slice_num) {
      glo_n_slice_num = pre_valid_loc_slice_n;
    }
    llvm::errs() << "pre_valid_loc_slice_n:" << pre_valid_loc_slice_n
                 << ", glo_n_slice_num:" << glo_n_slice_num
                 << ", pre_valid_loc_slice_c:" << pre_valid_loc_slice_c
                 << ", glo_c_slice_num:" << glo_c_slice_num
                 << ", pre_valid_loc_slice_h:" << pre_valid_loc_slice_h
                 << ", glo_h_slice_num:" << glo_h_slice_num << '\n';
    first_matmul = false;
  }

  llvm::errs() << "fill_slice_info start\n";
  lg_info.shape_secs.n_slice_num = glo_n_slice_num;
  lg_info.shape_secs.c_slice_num = glo_c_slice_num;
  lg_info.shape_secs.h_slice_num = glo_h_slice_num;
  fill_slice_info(lg_info);
  if (name() == "mlp_group" || name() == "single_matmul_group") {
    return true;
  }

  std::shared_ptr<CycleCalculator> cycle_calculator_;
  if (module::isCV18xx()) {
    Cv18xxCycleCalculator *cyc_ptr = new Cv18xxCycleCalculator();
    cycle_calculator_ = std::shared_ptr<CycleCalculator>(cyc_ptr);
  } else {
    Bm168xCycleCalculator *cyc_ptr = new Bm168xCycleCalculator();
    cycle_calculator_ = std::shared_ptr<CycleCalculator>(cyc_ptr);
  }

  tensor_info_t info;
  std::vector<std::pair<int, int>> vec_hslice_and_diff_cycle;
  int old_diff = -1, inc_time = 0;
  do {
    auto in = tmp_ops[0]->getOperand(1);
    if (lg_info.tensor_infos.find(in) != lg_info.tensor_infos.end()) {
      info = lg_info.tensor_infos[in];
    }
    info.mode2 = TIMESTEP2_LOAD;
    int load_cycle =
        cycle_calculator_->getGdmaCycle(in, info, lg_info._lgInfo.type);
    auto res = tmp_ops[0]->getResult(0);
    if (lg_info.tensor_infos.find(res) != lg_info.tensor_infos.end()) {
      info = lg_info.tensor_infos[res];
    }
    info.mode2 = TIMESTEP2_STORE;
    int store_cycle =
        cycle_calculator_->getGdmaCycle(res, info, lg_info._lgInfo.type);
    int bdc_cycle = cycle_calculator_->getLocalLayerCycle(
        tmp_ops[0], lg_info.tensor_infos, lg_info._lgInfo.type, true);
    auto diff = std::abs(bdc_cycle - store_cycle - load_cycle);
    llvm::errs() << "h_slice_num:" << lg_info.shape_secs.h_slice_num
                 << ", load_cycle:" << load_cycle
                 << ", store_cycle:" << store_cycle
                 << ", bdc_cycle:" << bdc_cycle << ", diff:" << diff << '\n';
    if (diff < old_diff && old_diff != -1) {
      inc_time = 0;
    } else {
      inc_time++;
      if (inc_time > 5) {
        llvm::errs() << "nc_time > 5, break\n";
        break;
      }
    }
    auto hslice_diff = std::make_pair(lg_info.shape_secs.h_slice_num, diff);
    vec_hslice_and_diff_cycle.push_back(hslice_diff);
    old_diff = diff;
    lg_info.shape_secs.h_slice_num++;
    fill_slice_info(lg_info);
  } while (true);
  std::sort(vec_hslice_and_diff_cycle.begin(), vec_hslice_and_diff_cycle.end(),
            pair_int_Sort_by_int);
  lg_info.shape_secs.h_slice_num = vec_hslice_and_diff_cycle[0].first;
  fill_slice_info(lg_info);
  llvm::errs() << "find best h_slice_num:" << lg_info.shape_secs.h_slice_num
               << ", diff:" << vec_hslice_and_diff_cycle[0].second << '\n';
  llvm::errs() << "n:" << lg_info.shape_secs.n << ", c:" << lg_info.shape_secs.c
               << ", h:" << lg_info.shape_secs.h
               << ", n_slice_num:" << lg_info.shape_secs.n_slice_num
               << ", glo_c_slice_num:" << glo_c_slice_num
               << ", glo_h_slice_num:" << glo_h_slice_num << '\n';
  return true;
}

static bool isElementwiseOp(Operation *op) {
  if (isa<tpu::ReshapeOp, tpu::ActiveOp, tpu::CastOp, tpu::MulConstOp,
          tpu::MulOp, tpu::AddOp, tpu::SubOp, tpu::DivOp>(op)) {
    return true;
  }
  if (isa<tpu::SliceOp>(op)) {
    return true;
  }
  return false;
}

static void CollectElementwiseOpAroundMatmul(
    Operation *op, const std::vector<Operation *> &subnet_ops,
    std::vector<Operation *> &result_ops, int forward_search = 0) {
  result_ops.push_back(op);
  if (forward_search == 0 || forward_search == 1) {
    for (auto v : op->getOperands()) {
      auto pre_op = v.getDefiningOp();
      if (std::find(subnet_ops.begin(), subnet_ops.end(), pre_op) !=
          subnet_ops.end()) {
        if (!isa<top::NoneOp, top::WeightOp, top::InputOp>(pre_op) &&
            isElementwiseOp(pre_op)) {
          llvm::errs() << "find pre_op " << module::getName(pre_op).str()
                       << "\n";
          CollectElementwiseOpAroundMatmul(pre_op, subnet_ops, result_ops, 1);
        }
      }
    }
  }

  if (forward_search == 0 || forward_search == 2) {
    for (auto user : op->getUsers()) {
      if (std::find(subnet_ops.begin(), subnet_ops.end(), user) !=
          subnet_ops.end()) {
        if (isElementwiseOp(user)) {
          llvm::errs() << "find user " << module::getName(user).str() << "\n";
          CollectElementwiseOpAroundMatmul(user, subnet_ops, result_ops, 2);
        }
      }
    }
  }
}

class single_matmul_group : public speical_layer_group_base {
public:
  virtual std::shared_ptr<speical_layer_group_base> clone() override {
    return std::make_shared<single_matmul_group>();
  }

  virtual bool
  pattern_match_and_parser(Operation *start_op,
                           std::vector<Operation *> &subnet_ops) override {
    if (isa<tpu::MatMulOp>(start_op)) {
      main_mm_op = start_op;
      llvm::errs() << "find single_matmul_group at "
                   << module::getName(start_op).str() << "\n";
      CollectElementwiseOpAroundMatmul(start_op, subnet_ops, ops);
      auto ops_reorder = sortOpsByOtherOpsOrder(subnet_ops, ops);
      ops.assign(ops_reorder.begin(), ops_reorder.end());

      for (auto op : ops) {
        if (isa<tpu::SliceOp>(op)) {
          int64_t in_n, in_c, in_d, in_h, in_w, out_n, out_c, out_d, out_h,
              out_w;
          module::getNCDHW(op->getOperand(0), in_n, in_c, in_d, in_h, in_w,
                           GROUP_MM_OPT3);
          module::getNCDHW(op->getResult(0), out_n, out_c, out_d, out_h, out_w,
                           GROUP_MM_OPT3);
          if (in_h != out_h) {
            col_cut = false;
            break;
          }
        }
      }

      std::vector<Operation *> del_ops, del_ops2;
      for (auto op : ops) {
        if (isa<tpu::ReshapeOp>(op)) {
          auto next_op = *(op->getUsers().begin());
          // reshape + matmul + reshape + (slice0 + slice1) -> matmul + (slice0
          // + slice1)
          if (isa<tpu::SliceOp, tpu::AddOp>(next_op) &&
              std::find(ops.begin(), ops.end(), next_op) != ops.end()) {
            auto mmOp = op->getOperand(0).getDefiningOp();
            if (mmOp && isa<tpu::MatMulOp>(mmOp)) {
              auto MMinReshapeOp =
                  dyn_cast<tpu::MatMulOp>(mmOp).getInput().getDefiningOp();
              if (MMinReshapeOp && isa<tpu::ReshapeOp>(MMinReshapeOp)) {
                bool has_out_user = false;
                for (auto user : MMinReshapeOp->getUsers()) {
                  if (find(ops.begin(), ops.end(), user) == ops.end()) {
                    has_out_user = true;
                    break;
                  }
                }
                if (!has_out_user) {
                  del_ops.push_back(MMinReshapeOp);
                }
                del_ops.push_back(op);
                auto oldType1 = MMinReshapeOp->getOperand(0).getType();
                auto oldType2 = op->getResult(0).getType();
                MMinReshapeOp->getResult(0).replaceUsesWithIf(
                    MMinReshapeOp->getOperand(0), [&](OpOperand &operand) {
                      Operation *user = operand.getOwner();
                      return find(ops.begin(), ops.end(), user) != ops.end();
                    });
                op->getResult(0).replaceUsesWithIf(
                    mmOp->getResult(0), [&](OpOperand &operand) {
                      Operation *user = operand.getOwner();
                      return find(ops.begin(), ops.end(), user) != ops.end();
                    });
                mmOp->getOperand(0).setType(oldType1);
                mmOp->getResult(0).setType(oldType2);
              }
            }
          }
        }
      }

      // for (auto op: ops) {
      //   if (isa<tpu::ReshapeOp>(op) && op->hasOneUse()) {
      //     auto next_op = *(op->getUsers().begin());
      //     // matmul + reshape + cast -> matmul + cast + reshape
      //     if (isa<tpu::CastOp>(next_op) && std::find(ops.begin(), ops.end(),
      //     next_op) != ops.end()) {
      //       auto mmOp = op->getOperand(0).getDefiningOp();
      //       if (mmOp && isa<tpu::MatMulOp>(mmOp) && mmOp->hasOneUse()) {
      //         auto MMinReshapeOp =
      //         dyn_cast<tpu::MatMulOp>(mmOp).getInput().getDefiningOp(); if
      //         (MMinReshapeOp && !isa<tpu::ReshapeOp>(MMinReshapeOp)) {
      //           auto oldType1 = op->getOperand(0).getType();
      //           auto oldType2 = op->getResult(0).getType();
      //           auto oldV1 = mmOp->getResult(0);
      //           auto oldV2 = op->getResult(0);
      //           auto oldV3 = next_op->getResult(0);
      //           next_op->getResult(0).replaceAllUsesWith(oldV2);
      //           mmOp->getResult(0).replaceAllUsesWith(oldV3);
      //           op->getResult(0).replaceAllUsesWith(oldV1);

      //           next_op->getOperand(0).setType(oldType1);
      //           next_op->getResult(0).setType(oldType1);
      //           op->getOperand(0).setType(oldType1);
      //           op->getResult(0).setType(oldType2);
      //           for (auto user: oldV3.getUsers()) {
      //             for (auto opd: user->getOperands()) {
      //               if (opd == oldV3) {
      //                 llvm::errs()<<"opd.setType\n";
      //                 opd.setType(oldType2);
      //               }
      //             }
      //           }
      //           llvm::errs()<<"new op:\n";
      //           next_op->dump();
      //           op->dump();
      //         }
      //       }
      //     }
      //   }
      // }

      findReshapeAtEdge(ops, del_ops2);
      for (auto del_op : del_ops2) {
        ops.erase(std::remove(ops.begin(), ops.end(), del_op), ops.end());
      }
      std::sort(del_ops.begin(), del_ops.end());
      auto last = std::unique(del_ops.begin(), del_ops.end());
      del_ops.erase(last, del_ops.end());
      for (auto del_op : del_ops) {
        llvm::errs() << "del_op: " << module::getName(del_op).str() << "\n";
        ops.erase(std::remove(ops.begin(), ops.end(), del_op), ops.end());
        need_del_ops.push_back(del_op);
      }
      std::vector<Operation *> break_ops, accessed_ops;
      find_op_tree_by_root2(start_op, h_cut_ops, ops, accessed_ops, break_ops);
      h_cut_ops.erase(std::remove(h_cut_ops.begin(), h_cut_ops.end(), start_op),
                      h_cut_ops.end());
      return check_group_valid();
    }
    return false;
  }

  virtual std::string name() override { return "single_matmul_group"; }
  virtual std::string brief() override { return "mlp in transformer block"; }
  virtual bool convert_to_other_type(
      std::vector<Operation *> &sub_ops,
      std::shared_ptr<speical_layer_group_base> &p_special_grp) override {
    if (!grp_is_valid(sub_ops)) {
      return false;
    }
    for (auto op : sub_ops) {
      if (isa<tpu::MatMulOp>(op)) {
        p_special_grp->main_mm_op = op;
        p_special_grp->h_cut_ops.assign(h_cut_ops.begin(), h_cut_ops.end());
        return true;
      }
    }
    return false;
  }
};

class mlp_group : public speical_layer_group_base {
public:
  virtual std::shared_ptr<speical_layer_group_base> clone() override {
    return std::make_shared<mlp_group>();
  }

  virtual bool
  pattern_match_and_parser(Operation *start_op,
                           std::vector<Operation *> &subnet_ops) override {
    Operation *next_matmul_op = nullptr;
    if (search_two_mmOp(start_op, next_matmul_op, subnet_ops)) {
      auto mmOp = dyn_cast<tpu::MatMulOp>(next_matmul_op);
      if (dyn_cast<tpu::MatMulOp>(start_op).getHdimIsBatch() ||
          mmOp.getHdimIsBatch()) {
        return false; // matmul with HdimIsBatch is not included
      }
      map_value_to_cut_dims[mmOp.getRight()] = {0, 3, 2, 1, 4};
      std::vector<Operation *> del_ops;
      for (auto op : ops) {
        if (isa<tpu::ActiveOp>(op)) {
          auto next_op = *(op->getUsers().begin());
          if (isa<tpu::ReshapeOp>(next_op) &&
              std::find(ops.begin(), ops.end(), next_op) != ops.end()) {
            assert((op->getResult(0).hasOneUse()));
            bool has_out_user = false;
            for (auto user : next_op->getUsers()) {
              if (find(ops.begin(), ops.end(), user) == ops.end()) {
                has_out_user = true;
                break;
              }
            }
            if (!has_out_user) {
              del_ops.push_back(next_op);
            }
            op->getResult(0).setType(next_op->getResult(0).getType());
            next_op->getResult(0).replaceUsesWithIf(
                op->getResult(0), [&](OpOperand &operand) {
                  Operation *user = operand.getOwner();
                  return find(ops.begin(), ops.end(), user) != ops.end();
                });
          }

          auto inOp = op->getOperand(0).getDefiningOp();
          if (inOp && isa<tpu::ReshapeOp>(inOp)) {
            bool has_out_user = false;
            for (auto user : inOp->getUsers()) {
              if (find(ops.begin(), ops.end(), user) == ops.end()) {
                has_out_user = true;
                break;
              }
            }
            if (!has_out_user) {
              del_ops.push_back(inOp);
            }
            auto oldType = inOp->getOperand(0).getType();
            inOp->getResult(0).replaceUsesWithIf(
                inOp->getOperand(0), [&](OpOperand &operand) {
                  Operation *user = operand.getOwner();
                  return find(ops.begin(), ops.end(), user) != ops.end();
                });
            op->getOperand(0).setType(oldType);
          }
        }
      }

      // for (auto op: ops) {
      //   if (isa<tpu::MatMulOp>(op)) {
      //     auto next_op = *(op->getUsers().begin());
      //     if (isa<tpu::ReshapeOp>(next_op) && std::find(ops.begin(),
      //     ops.end(), next_op) != ops.end()) {
      //       assert((op->getResult(0).hasOneUse()));
      //       bool has_out_user = false;
      //       for (auto user: next_op->getUsers()) {
      //         if (find(ops.begin(), ops.end(), user) == ops.end()) {
      //           has_out_user = true;
      //           break;
      //         }
      //       }
      //       if (!has_out_user) {
      //         del_ops.push_back(next_op);
      //       }
      //       op->getResult(0).setType(next_op->getResult(0).getType());
      //       next_op->getResult(0).replaceUsesWithIf(op->getResult(0),
      //       [&](OpOperand &operand) {
      //         Operation *user = operand.getOwner();
      //         return find(ops.begin(), ops.end(), user) != ops.end();
      //       });
      //     } else {
      //       if (op == ops[0]) {
      //         break;
      //       }
      //     }

      //     auto inOp = dyn_cast<tpu::MatMulOp>(op).getInput().getDefiningOp();
      //     if (inOp && isa<tpu::ReshapeOp>(inOp)) {
      //       bool has_out_user = false;
      //       for (auto user: inOp->getUsers()) {
      //         if (find(ops.begin(), ops.end(), user) == ops.end()) {
      //           has_out_user = true;
      //           break;
      //         }
      //       }
      //       if (!has_out_user) {
      //         del_ops.push_back(inOp);
      //       }
      //       auto oldType = inOp->getOperand(0).getType();
      //       inOp->getResult(0).replaceUsesWithIf(inOp->getOperand(0),
      //       [&](OpOperand &operand) {
      //         Operation *user = operand.getOwner();
      //         return find(ops.begin(), ops.end(), user) != ops.end();
      //       });
      //       op->getOperand(0).setType(oldType);
      //     }
      //   }
      // }

      for (auto del_op : del_ops) {
        need_del_ops.push_back(del_op);
        ops.erase(std::remove(ops.begin(), ops.end(), del_op), ops.end());
      }

      for (auto op : ops) {
        if (isa<tpu::ConcatOp>(op)) {
          auto in_shape = module::getShapeVec(op->getOperand(0));
          if (in_shape.size() == dyn_cast<tpu::ConcatOp>(op).getAxis() + 1) {
            col_cut = false;
          }
        }
      }
      return ops.size() > 1 && check_group_valid();
    }
    return false;
  }

  virtual std::string name() override { return "mlp_group"; }
  virtual std::string brief() override { return "mlp in transformer block"; }

  virtual bool convert_to_other_type(
      std::vector<Operation *> &sub_ops,
      std::shared_ptr<speical_layer_group_base> &p_special_grp) override {
    if (!grp_is_valid(sub_ops)) {
      return false;
    }
    if (isa<tpu::MatMulOp>(sub_ops[0])) {
      ops.assign(sub_ops.begin(), sub_ops.end());
    } else {
      p_special_grp = std::make_shared<single_matmul_group>();
      p_special_grp->ops.assign(sub_ops.begin(), sub_ops.end());
      p_special_grp->main_mm_op = sub_ops.back();
    }
    return true;
  }
};

class attention_group : public speical_layer_group_base {
public:
  virtual std::shared_ptr<speical_layer_group_base> clone() override {
    return std::make_shared<attention_group>();
  }

  virtual bool
  pattern_match_and_parser(Operation *start_op,
                           std::vector<Operation *> &subnet_ops) override {
    Operation *next_matmul_op = nullptr;
    if (module::isDebugCmdEnable("print_mutmul_group_match_process")) {
      llvm::errs() << "start match attention_group in ops:\n";
      for (auto it : subnet_ops) {
        llvm::errs() << show_op_info(it) << "\n";
      }
    }
    if (search_two_mmOp(start_op, next_matmul_op, subnet_ops)) {
      llvm::errs() << "find_softmax:" << find_softmax << '\n';
      return find_softmax && check_group_valid();
    }
    return false;
  }
  virtual std::string name() override { return "attention_group"; }

  virtual bool CalcMatMulGroupTpNum(ilp_LgInfo &lg_info, Operation *&failed_op,
                                    int64_t core_num) override {
    auto mm_op = dyn_cast<tpu::MatMulOp>(main_mm_op);
    if (mm_op && !mm_op.getHdimIsBatch()) {
      return speical_layer_group_base::CalcMatMulGroupTpNum(lg_info, failed_op,
                                                            core_num);
    }
    hdim_is_batch = true;

    int64_t cut_n, in_c, cut_h, in_w, out_c, out_w;
    group_type_t type = lg_info._lgInfo.type;
    lg_info.p_special_grp->get_batch_size(lg_info.shape_secs);
    int batch_size = lg_info.shape_secs.n;
    int glo_n_slice_num = get_best_n_slice_num(batch_size, core_num),
        glo_c_slice_num = 1, glo_h_slice_num = 1;
    bool enable_cut_h = false;

    std::vector<Operation *> tmp_ops, tmp_ops2;
    for (auto op : lg_info._lgInfo.group_ops) {
      if (isa<tpu::MatMulOp>(op)) {
        tmp_ops.push_back(op);
      } else {
        tmp_ops2.push_back(op);
      }
    }
    if (tmp_ops.size() == 2) {
      std::swap(tmp_ops[0], tmp_ops[1]);
    }
    tmp_ops.insert(tmp_ops.end(), tmp_ops2.begin(), tmp_ops2.end());
    bool first_matmul = true;
    for (auto op : tmp_ops) {
      auto ins = get_input_values(op);
      auto outs = get_output_values(op);
      int try_n_slice_num = glo_n_slice_num, try_c_slice_num = glo_c_slice_num,
          try_h_slice_num = glo_h_slice_num;
      int pre_valid_loc_slice_n = glo_n_slice_num,
          pre_valid_loc_slice_h = glo_h_slice_num,
          pre_valid_loc_slice_c = glo_c_slice_num;
      llvm::errs() << "CalcMatMulGroupTpNum for op:"
                   << module::getName(op).str() << '\n';
      int secs =
          get_secs(op, try_n_slice_num, try_c_slice_num, try_h_slice_num);
      int old_target_secs = align(secs, core_num);
      bool init_secs_is_ok = old_target_secs == secs;
      do {
        auto shape = ins[0].getType().cast<RankedTensorType>().getShape().vec();
        ;
        if (pre_valid_loc_slice_n > shape[0]) {
          cut_n = 1;
          cut_h = ceiling_func(shape[2], pre_valid_loc_slice_n / shape[0]);
        } else {
          cut_n = ceiling_func(shape[0], pre_valid_loc_slice_n);
          cut_h = shape[2];
        }
        in_w = shape[3];
        int64_t in0_lmem_bytes = 0, in1_lmem_bytes = 0, out0_lmem_bytes = 0;
        in_c = align(shape[1], try_c_slice_num) / try_c_slice_num;
        in0_lmem_bytes = align_64(
            Arch::get_tensor_lmem_bytes(ins[0], cut_n, in_c, 1, cut_h, in_w));

        shape = outs[0].getType().cast<RankedTensorType>().getShape().vec();
        ;
        out_c = align(shape[1], try_c_slice_num) / try_c_slice_num;
        out_w = shape[3];
        if (isa<tpu::MatMulOp>(op) && op == ops.back()) {
          out_w = align(out_w, try_h_slice_num) / try_h_slice_num;
        }
        out0_lmem_bytes = align_64(Arch::get_tensor_lmem_bytes(
            outs[0], cut_n, out_c, 1, cut_h, out_w));
        if (isa<tpu::MatMulOp>(op) && try_h_slice_num > 1 && op == ops.back()) {
          // The second matmul input and output occupy twice the memory to
          // consider the store of the previous time slot and the load of the
          // next time slot
          out0_lmem_bytes *= 2;
        }

        auto lg_op = cast<LocalGenInterface>(op);
        int64_t buffer_size = lg_op.getBufferSize(
            in0_lmem_bytes, out0_lmem_bytes, cut_n, in_c, cut_h, 1, in_w, cut_n,
            out_c, cut_h, 1, out_w, type);
        if (ins.size() > 1) {
          shape = ins[1].getType().cast<RankedTensorType>().getShape().vec();
          ;
          if (isa<tpu::MatMulOp>(op) && op == ops.back()) {
            auto mm_op = dyn_cast<tpu::MatMulOp>(op);
            if (mm_op.getRightTranspose()) {
              shape[1] = align(shape[1], try_h_slice_num) / try_h_slice_num;
            } else {
              shape[3] = align(shape[3], try_h_slice_num) / try_h_slice_num;
            }
          }
          in1_lmem_bytes = align_64(Arch::get_tensor_lmem_bytes(
              ins[1], cut_n, shape[1], 1, cut_h, shape[3]));
          if (isa<tpu::MatMulOp>(op)) {
            if (op == ops.back()) {
              if (try_h_slice_num > 1) {
                in1_lmem_bytes *= 2;
              }
            } else {
              if (in1_lmem_bytes >= Arch::LMEM_BYTES) {
                llvm::errs() << "the right matrix of Matmul before softmax is "
                                "too large, op:"
                             << module::getName(op).str() << "\n";
                failed_op = op;
                return false;
              }
            }
          }
        }
        bool inc_c_slice = true;
        if (isa<tpu::MatMulOp>(op) && op == ops.back()) {
          inc_c_slice = in0_lmem_bytes >= in1_lmem_bytes; //相等时优先切c
        }

        int total =
            buffer_size + in0_lmem_bytes + in1_lmem_bytes + out0_lmem_bytes;
        bool mem_enough = total <= Arch::LMEM_BYTES;
        llvm::errs() << "in0_lmem_bytes:" << in0_lmem_bytes
                     << ", out0_lmem_bytes:" << out0_lmem_bytes
                     << ", in1_lmem_bytes:" << in1_lmem_bytes
                     << ", buffer_size:" << buffer_size << ", total:" << total
                     << ", inc_c_slice:" << inc_c_slice
                     << ", old_target_secs:" << old_target_secs
                     << ", mem_enough:" << mem_enough << '\n';
        if (mem_enough) {
          if (init_secs_is_ok || !first_matmul) { // todo why first_matmul?
            llvm::errs() << "init_secs_is_ok\n";
            break;
          }
          if (!inc_slice_num(op, try_n_slice_num, try_c_slice_num,
                             try_h_slice_num, batch_size, lg_info.shape_secs.c,
                             enable_cut_h ? lg_info.shape_secs.h : 1,
                             old_target_secs, inc_c_slice)) {
            failed_op = op;
            return false;
          }
          secs =
              get_secs(op, try_n_slice_num, try_c_slice_num, try_h_slice_num);
          if (secs > old_target_secs) {
            llvm::errs() << "new secs(" << secs
                         << ") >= old_target_secs, break\n";
            break;
          }
        } else {
          if (!inc_slice_num(op, try_n_slice_num, try_c_slice_num,
                             try_h_slice_num, batch_size, lg_info.shape_secs.c,
                             enable_cut_h ? lg_info.shape_secs.h : 1,
                             old_target_secs, inc_c_slice)) {
            failed_op = op;
            return false;
          }
          secs =
              get_secs(op, try_n_slice_num, try_c_slice_num, try_h_slice_num);
          old_target_secs = align(secs, core_num);
          init_secs_is_ok = old_target_secs == secs;
        }
        pre_valid_loc_slice_n = try_n_slice_num;
        pre_valid_loc_slice_c = try_c_slice_num;
        pre_valid_loc_slice_h = try_h_slice_num;
      } while (true);

      if (pre_valid_loc_slice_c > glo_c_slice_num) {
        glo_c_slice_num = pre_valid_loc_slice_c;
      }
      if (pre_valid_loc_slice_h > glo_h_slice_num) {
        glo_h_slice_num = pre_valid_loc_slice_h;
      }
      if (pre_valid_loc_slice_n > glo_n_slice_num) {
        glo_n_slice_num = pre_valid_loc_slice_n;
      }
      llvm::errs() << "pre_valid_loc_slice_n:" << pre_valid_loc_slice_n
                   << ", glo_n_slice_num:" << glo_n_slice_num
                   << ", pre_valid_loc_slice_c:" << pre_valid_loc_slice_c
                   << ", glo_c_slice_num:" << glo_c_slice_num
                   << ", pre_valid_loc_slice_h:" << pre_valid_loc_slice_h
                   << ", glo_h_slice_num:" << glo_h_slice_num << '\n';
      first_matmul = false;
    }

    llvm::errs() << "fill_slice_info start\n";
    lg_info.shape_secs.n_slice_num = glo_n_slice_num;
    lg_info.shape_secs.c_slice_num = glo_c_slice_num;
    lg_info.shape_secs.h_slice_num = glo_h_slice_num;
    fill_slice_info(lg_info);
    if (!enable_cut_h) {
      return true;
    }
    std::shared_ptr<CycleCalculator> cycle_calculator_;
    if (module::isCV18xx()) {
      Cv18xxCycleCalculator *cyc_ptr = new Cv18xxCycleCalculator();
      cycle_calculator_ = std::shared_ptr<CycleCalculator>(cyc_ptr);
    } else {
      Bm168xCycleCalculator *cyc_ptr = new Bm168xCycleCalculator();
      cycle_calculator_ = std::shared_ptr<CycleCalculator>(cyc_ptr);
    }

    tensor_info_t info;
    std::vector<std::pair<int, int>> vec_hslice_and_diff_cycle;
    int old_diff = -1, inc_time = 0;
    do {
      auto in = tmp_ops[0]->getOperand(1);
      if (lg_info.tensor_infos.find(in) != lg_info.tensor_infos.end()) {
        info = lg_info.tensor_infos[in];
      }
      info.mode2 = TIMESTEP2_LOAD;
      int load_cycle =
          cycle_calculator_->getGdmaCycle(in, info, lg_info._lgInfo.type);
      auto res = tmp_ops[0]->getResult(0);
      if (lg_info.tensor_infos.find(res) != lg_info.tensor_infos.end()) {
        info = lg_info.tensor_infos[res];
      }
      info.mode2 = TIMESTEP2_STORE;
      int store_cycle =
          cycle_calculator_->getGdmaCycle(res, info, lg_info._lgInfo.type);
      int bdc_cycle = cycle_calculator_->getLocalLayerCycle(
          tmp_ops[0], lg_info.tensor_infos, lg_info._lgInfo.type, true);
      auto diff = std::abs(bdc_cycle - store_cycle - load_cycle);
      llvm::errs() << "h_slice_num:" << lg_info.shape_secs.h_slice_num
                   << ", load_cycle:" << load_cycle
                   << ", store_cycle:" << store_cycle
                   << ", bdc_cycle:" << bdc_cycle << ", diff:" << diff << '\n';
      if (diff < old_diff && old_diff != -1) {
        inc_time = 0;
      } else {
        inc_time++;
        if (inc_time > 5) {
          llvm::errs() << "nc_time > 5, break\n";
          break;
        }
      }
      auto hslice_diff = std::make_pair(lg_info.shape_secs.h_slice_num, diff);
      vec_hslice_and_diff_cycle.push_back(hslice_diff);
      old_diff = diff;
      lg_info.shape_secs.h_slice_num++;
      fill_slice_info(lg_info);
    } while (true);
    std::sort(vec_hslice_and_diff_cycle.begin(),
              vec_hslice_and_diff_cycle.end(), pair_int_Sort_by_int);
    lg_info.shape_secs.h_slice_num = vec_hslice_and_diff_cycle[0].first;
    fill_slice_info(lg_info);
    llvm::errs() << "find best h_slice_num:" << lg_info.shape_secs.h_slice_num
                 << ", diff:" << vec_hslice_and_diff_cycle[0].second << '\n';
    llvm::errs() << "n:" << lg_info.shape_secs.n
                 << ", c:" << lg_info.shape_secs.c
                 << ", h:" << lg_info.shape_secs.h
                 << ", n_slice_num:" << lg_info.shape_secs.n_slice_num
                 << ", glo_c_slice_num:" << glo_c_slice_num
                 << ", glo_h_slice_num:" << glo_h_slice_num << '\n';
    return true;
  }

  virtual bool convert_to_other_type(
      std::vector<Operation *> &sub_ops,
      std::shared_ptr<speical_layer_group_base> &p_special_grp) override {
    if (!grp_is_valid(sub_ops)) {
      return false;
    }
    bool have_softmax = false;
    Operation *mm_op = nullptr;
    for (auto op : sub_ops) {
      if (op && isa<tpu::SoftmaxOp>(op)) {
        have_softmax = true;
      }
      if (op && isa<tpu::MatMulOp>(op)) {
        mm_op = op;
      }
    }

    if (sub_ops.back() && isa<tpu::MatMulOp>(sub_ops.back())) {
      if (have_softmax) {
        ops.assign(sub_ops.begin(), sub_ops.end());
        p_special_grp->main_mm_op = mm_op;
        return true;
      }
    } else {
      if (sub_ops.front() && isa<tpu::MatMulOp>(sub_ops.front())) {
        p_special_grp = std::make_shared<single_matmul_group>();
        p_special_grp->ops.assign(sub_ops.begin(), sub_ops.end());
        p_special_grp->main_mm_op = mm_op;
        return true;
      }
    }
    return false;
  }

  virtual std::string brief() override { return "mlp in transformer block"; }
};

void GroupOps::SearchGroup(std::vector<dag_subnet> &dag_subnets,
                           std::shared_ptr<speical_layer_group_base> grp_ptr) {
  while (true) {
    bool all_checked = true;
    bool dag_subnets_update = false;
    std::vector<dag_subnet> new_dag_subnets;
    for (auto &dag : dag_subnets) {
      if (!dag.checked) {
        all_checked = false;
        for (auto op : dag.ops) {
          auto tmp_grp_ptr = grp_ptr->clone();
          if (tmp_grp_ptr->pattern_match_and_parser(op, dag.ops)) {
            if (!grp_is_valid(tmp_grp_ptr->ops)) {
              continue;
            }
            need_del_ops.insert(need_del_ops.end(),
                                tmp_grp_ptr->need_del_ops.begin(),
                                tmp_grp_ptr->need_del_ops.end());
            dag.matched = true;
            auto lgInfo = CreateIlpLgInfo(tmp_grp_ptr->ops, options_);
            lgInfo->p_special_grp = tmp_grp_ptr;
            lgInfo->_lgInfo.type = GROUP_MM_OPT3;
            lg_pass_ir_->tmp_base_groups.push_back(lgInfo);

            for (auto grp_ops2 :
                 seg_network_by_group_ops(dag.ops, tmp_grp_ptr->ops)) {
              if (grp_is_valid(grp_ops2)) {
                dag_subnet tmp_dag;
                tmp_dag.ops.assign(grp_ops2.begin(), grp_ops2.end());
                new_dag_subnets.push_back(tmp_dag);
                dag_subnets_update = true;
              }
            }
            llvm::errs() << "find a " << tmp_grp_ptr->name() << ":\n";
            for (auto it : tmp_grp_ptr->ops) {
              llvm::errs() << show_op_info(it) << "\n";
            }
            break;
          }
        }
        dag.checked = true;
        if (dag_subnets_update) {
          break;
        }
      }
    }
    dag_subnets.insert(dag_subnets.end(), new_dag_subnets.begin(),
                       new_dag_subnets.end());
    if (all_checked) {
      break;
    }
  }

  std::vector<dag_subnet> tmp_dag_subnets;
  for (auto dag : dag_subnets) {
    if (!dag.matched) {
      dag.checked = false;
      tmp_dag_subnets.push_back(dag);
    }
  }
  dag_subnets.assign(tmp_dag_subnets.begin(), tmp_dag_subnets.end());
}

void GroupOps::findSpecialGroup(llvm::SetVector<Operation *> &subnet_ops) {
  std::vector<Operation *> tmp_ops, excluded_ops;
  std::vector<dag_subnet> all_dag_subnet;

  dag_subnet tmp_dag;
  tmp_dag.ops.assign(subnet_ops.begin(), subnet_ops.end());
  all_dag_subnet.push_back(tmp_dag);

  llvm::errs() << "find attention_group\n";
  SearchGroup(all_dag_subnet, std::make_shared<attention_group>());
  llvm::errs() << "find mlp_group\n";
  SearchGroup(all_dag_subnet, std::make_shared<mlp_group>());
  llvm::errs() << "find single_matmul_group\n";
  SearchGroup(all_dag_subnet, std::make_shared<single_matmul_group>());

  for (auto dag : all_dag_subnet) {
    if (grp_is_valid(dag.ops)) {
      std::vector<Operation *> global_layers;
      for (auto op : dag.ops) {
        if (!isa<ReturnOp, FuncOp, top::NoneOp, top::WeightOp, top::InputOp>(
                op) &&
            !isLgSupport(op)) {
          global_layers.push_back(op);
        }
      }
      excluded_ops.clear();
      for (auto ops2 : seg_grp_ops_by_global_op(dag.ops, global_layers,
                                                excluded_ops, options_)) {
        if (grp_is_valid(ops2)) {
          llvm::errs() << "normal group, ops2.size:" << ops2.size() << "\n";
          lg_pass_ir_->tmp_base_groups.push_back(
              CreateIlpLgInfo(ops2, options_));
        }
      }
    }
  }

  for (auto op : need_del_ops) {
    auto it = std::find(subnet_ops.begin(), subnet_ops.end(), op);
    if (it != subnet_ops.end()) {
      subnet_ops.erase(it);
      op->erase();
    }
  }
}

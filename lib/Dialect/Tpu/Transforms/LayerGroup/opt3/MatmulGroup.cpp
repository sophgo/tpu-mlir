//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <fstream>
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/InternalOptimizer.h"
#include "tpu_mlir/Support/TopoSorter.h"

using namespace tpu_mlir::tpu;
using namespace tpu_mlir::backend;

bool pair_int_Sort_by_int(const std::pair<int, int> &v1,
                             const std::pair<int, int> &v2) {
  return v1.second < v2.second;
}

void backward_collect_op(const Value &out, std::list<Value> &tensor_branchs, std::vector<Operation*>& last_op_all_pre_ops) {
  auto op = out.getDefiningOp();
  if (op) {
    for (auto in : op->getOperands()) {
      auto pre_op = in.getDefiningOp();
      if (pre_op != nullptr && isa<top::NoneOp, top::WeightOp, top::InputOp>(pre_op)) {
        continue;
      }
      last_op_all_pre_ops.push_back(pre_op);
      tensor_branchs.push_back(in);
    }
  }
}

//start_op前面多个层类型为pre_op_names，单链匹配
bool matchPreOpName(Operation* start_op, std::vector<std::string> pre_op_names) {
  for (auto pre_op_name: pre_op_names) {
    int pre_op_count = 0;
    for (auto it: start_op->getOperands()) {
      auto pre_op = it.getDefiningOp();
      if (pre_op && !isa<top::NoneOp, top::WeightOp>(pre_op)) {
        pre_op_count++; //统计前驱op个数
      }
    }
    if (pre_op_count == 1) {
      auto pre_op2 = start_op->getOperand(0).getDefiningOp();
      if (pre_op2->getName().getStringRef().str() != pre_op_name) {
        return false;
      } else {
        start_op = pre_op2;
      }
    } else {
      return false;
    }
  }
  return true;
}

void DfsFindNextMatMul(Operation* start_op, Operation*& next_matmul_op) {
  for (auto user: start_op->getUsers()) {
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

void BfsFindNextMatMul(Operation* start_op, Operation*& next_matmul_op) {
  std::queue<Operation*> q;
  for (auto user: start_op->getUsers()) {
    if (!isa<ReturnOp>(user)) {
      q.push(user);
    }
  }

  while(!q.empty()) {
    auto op = q.front();
    q.pop();
    if (isa<tpu::MatMulOp>(op)) {
      next_matmul_op = op;
      return;
    }
    if (isa<tpu::SoftmaxOp>(op) || module::isInMatMulGrpOp(op)) {
      for (auto user: op->getUsers()) {
        if (!isa<ReturnOp>(user)) {
          q.push(user);
        }
      }
    }
  }
}

void FindValidOpUntilMatMul(Operation* start_op, std::vector<Operation*>& find_ops,
  std::vector<Operation*>& next_matmul_pre_ops, bool& find_softmax, bool& failed) {
  for (auto user: start_op->getUsers()) {
    if (isa<ReturnOp>(user)) {
      continue;
    }
    if (std::find(next_matmul_pre_ops.begin(), next_matmul_pre_ops.end(), user)
          == next_matmul_pre_ops.end()) {
      continue;
    }
    if (isa<tpu::MatMulOp>(user)) {
      if (std::find(find_ops.begin(), find_ops.end(), user) == find_ops.end()) {
        find_ops.push_back(user);
      }
      return;
    }

    if (isa<tpu::SoftmaxOp>(user)) {
      find_softmax = true;
    }
    if (isa<tpu::SoftmaxOp>(user) || module::isInMatMulGrpOp(user)) {
      if (std::find(find_ops.begin(), find_ops.end(), user) == find_ops.end()) {
        llvm::errs() <<"find op:"<<module::getName(user).str()<<"\n";
        find_ops.push_back(user);
        FindValidOpUntilMatMul(user, find_ops, next_matmul_pre_ops, find_softmax, failed);
      }
    } else {
      llvm::errs() <<"fail2 at op:"<<module::getName(user).str()<<"\n";
      failed = true;
      return;
    }
  }
}

void updateReshapeNextOpInValue(Operation* ReshapeOp, std::vector<Operation*>& grp_ops,
std::vector<Operation*>& del_ops, std::vector<Operation*>& subnet_ops, bool reshape_after_matmal = false) {
  bool has_grp_out = false;
  auto reshape_out = ReshapeOp->getResult(0);
  for (auto user: ReshapeOp->getUsers()) {
    if (std::find(grp_ops.begin(), grp_ops.end(), user) != grp_ops.end()) {
      for (OpOperand &opd : user->getOpOperands()) {
        auto pre_op = opd.get().getDefiningOp();
        if (pre_op && isa<top::NoneOp>(pre_op)) {
          continue;
        }
        if (reshape_out == opd.get()) {
          user->setOperand(opd.getOperandNumber(), ReshapeOp->getOperand(0));
          if (reshape_after_matmal) {
            user->getOperand(0).setType(ReshapeOp->getOperand(0).getType());
          } else {
            ReshapeOp->getOperand(0).setType(reshape_out.getType());
          }
        }
      }
    } else {
      has_grp_out = true;
    }
  }

  del_ops.push_back(ReshapeOp);
  if (!has_grp_out) {
    subnet_ops.erase(std::remove(subnet_ops.begin(), subnet_ops.end(), ReshapeOp), subnet_ops.end());
    // ReshapeOp->erase();
  }
}

bool speical_layer_group_base::search_two_mmOp(Operation* start_op, Operation*& next_mmOp,
                                               std::vector<Operation*>& subnet_ops,
                                               std::vector<Operation*>& accessed_ops) {
  if (isa<tpu::MatMulOp>(start_op)) {
    next_mmOp = nullptr;
    BfsFindNextMatMul(start_op, next_mmOp);
    if (next_mmOp) {
      std::vector<Operation*> pre_ops;
      find_all_pre_ops(next_mmOp, pre_ops, &subnet_ops);
      ops.clear();
      bool failed = false;
      FindValidOpUntilMatMul(start_op, ops, pre_ops, find_softmax, failed);
      if (!failed) {
        ops.push_back(start_op);
        auto ops_reorder = sortOpsByOtherOpsOrder(subnet_ops, ops);
        ops.assign(ops_reorder.begin(), ops_reorder.end());
        return isCheckOpInOtherOps(ops, accessed_ops);
      }
    }
  }
  return false;
}

void speical_layer_group_base::get_batch_size(shape_secs_t &shape_secs) {
  int64_t in_n, in_c, in_d, in_h, in_w;
  Operation* op;
  if (name() == "mlp_group") {
    op = ops[0];
  } else if (name() == "attention_group") {
    op = ops.back();
  } else {
    assert(false);
  }
  module::getNCDHW(op->getOperand(0), in_n, in_c, in_d, in_h, in_w, GROUP_MM_OPT3);
  shape_secs.n = in_n;
  shape_secs.c = in_c;
  if (dyn_cast<tpu::MatMulOp>(op).getLeftTranspose()) {
    shape_secs.c = in_h;
  }
  module::getNCDHW(op->getOperand(1), in_n, in_c, in_d, in_h, in_w, GROUP_MM_OPT3);
  shape_secs.h = in_h;
  if (dyn_cast<tpu::MatMulOp>(op).getRightTranspose()) {
    shape_secs.h = in_c;
  }
  if (!col_cut) {
    shape_secs.h = 1;
  }
  llvm::errs() << "get matmul group n:" <<shape_secs.n<< ", c:" <<shape_secs.c<< ", h:" <<shape_secs.h<<'\n';
}

bool speical_layer_group_base::update_shape_secs_for_ilp_group(shape_secs_t &shape_secs,const shape_secs_t &max_shape_secs) {
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
  llvm::errs() <<"n_slice_num: "<<ilp_lg_info.shape_secs.n_slice_num
                <<", c_slice_num: "<<ilp_lg_info.shape_secs.c_slice_num
                <<", h_slice_num: "<<ilp_lg_info.shape_secs.h_slice_num<<"\n";
  for (auto op: ilp_lg_info._lgInfo.group_ops) {
    for (auto in: get_input_values(op)) {
      module::getNCDHW(in, n, c, d, h, w, ilp_lg_info._lgInfo.type);
      slice_info_t si;
      slice_distributor(si.n, n, ilp_lg_info.shape_secs.n_slice_num);
      slice_distributor(si.c, c, 1);
      slice_distributor(si.d, d, 1);
      slice_distributor(si.h, h, 1);
      slice_distributor(si.w, w, 1);
      if (module::IsRightMat(in)) {
        if (name() == "mlp_group") {
          if (ilp_lg_info._lgInfo.group_ops[0] == op) {
            llvm::errs() <<"in: "<<module::getName(in).str()<<", cut h to h_slice_num\n";
            slice_distributor(si.h, h, ilp_lg_info.shape_secs.h_slice_num);
          } else {
            llvm::errs() <<"in: "<<module::getName(in).str()<<", cut c to h_slice_num\n";
            slice_distributor(si.c, c, ilp_lg_info.shape_secs.h_slice_num);
          }
        } else if (name() == "attention_group") {
          if (ilp_lg_info._lgInfo.group_ops.back() == op) {
            llvm::errs() <<"in: "<<module::getName(in).str()<<", cut h to h_slice_num\n";
            slice_distributor(si.h, h, ilp_lg_info.shape_secs.h_slice_num);
          }
        }
        ilp_lg_info.value_load_to_l2m[in] = -1;
      } else {
        slice_distributor(si.c, c, ilp_lg_info.shape_secs.c_slice_num);
        llvm::errs() <<"in: "<<module::getName(in).str()<<", cut c to c_slice_num\n";
        if (name() == "mlp_group") {
          if (isa<tpu::MatMulOp>(op)) {
            if (ilp_lg_info._lgInfo.group_ops[0] != op) {
              slice_distributor(si.h, h, ilp_lg_info.shape_secs.h_slice_num);
              llvm::errs() <<"in: "<<module::getName(in).str()<<", cut h to h_slice_num\n";
            }
          } else {
            slice_distributor(si.h, h, ilp_lg_info.shape_secs.h_slice_num);
          }
        }
      }
      ilp_lg_info.tensor_infos[in] = tensor_info_t(si);
    }

    for (auto out: get_output_values(op)) {
      module::getNCDHW(out, n, c, d, h, w, ilp_lg_info._lgInfo.type);
      llvm::errs() <<"out: "<<module::getName(out).str()<<", cut n/c to n/c_slice_num\n";
      slice_info_t si;
      slice_distributor(si.n, n, ilp_lg_info.shape_secs.n_slice_num);
      slice_distributor(si.c, c, ilp_lg_info.shape_secs.c_slice_num);
      slice_distributor(si.d, d, 1);
      slice_distributor(si.h, h, 1);
      slice_distributor(si.w, w, 1);
      if (name() == "mlp_group") {
        if (ilp_lg_info._lgInfo.group_ops.back() != op) {
          llvm::errs() <<"out: "<<module::getName(out).str()<<", cut h to h_slice_num\n";
          slice_distributor(si.h, h, ilp_lg_info.shape_secs.h_slice_num);
        } else {
          if (ilp_lg_info.shape_secs.h_slice_num > 1) {
            ilp_lg_info.value_store_to_l2m[out] = -1;
          }
        }
      } else if (name() == "attention_group") {
        if (ilp_lg_info._lgInfo.group_ops.back() == op) {
          llvm::errs() <<"out: "<<module::getName(out).str()<<", cut h to h_slice_num\n";
          slice_distributor(si.h, h, ilp_lg_info.shape_secs.h_slice_num);
        }
      }
      ilp_lg_info.tensor_infos[out] = tensor_info_t(si);
    }
  }
}


bool speical_layer_group_base::inc_slice_num(int& test_slice_n, int& try_c_slice_num, int& try_h_slice_num,
                    int max_c_slice_num, int max_h_slice_num,  bool inc_c_slice) {
  if (test_slice_n > 1) {
    llvm::errs() << "test_slice_n--\n";
    test_slice_n--;
  } else {
    if (inc_c_slice) {
      if (try_c_slice_num < max_c_slice_num) {
        llvm::errs() << "try_c_slice_num++\n";
        try_c_slice_num++;
      } else {
        if (try_h_slice_num < max_h_slice_num) {
          llvm::errs() << "try_h_slice_num++\n";
          try_h_slice_num++;
        } else {
          llvm::errs() << "inc_slice_num fail1\n";
          return false;
        }
      }
    } else {
      if (try_h_slice_num < max_h_slice_num) {
        llvm::errs() << "try_h_slice_num++\n";
        try_h_slice_num++;
      } else {
        if (try_c_slice_num < max_c_slice_num) {
          llvm::errs() << "try_c_slice_num++\n";
          try_c_slice_num++;
        } else {
          llvm::errs() << "inc_slice_num fail2\n";
          return false;
        }
      }
    }
  }
  return true;
}

int speical_layer_group_base::get_secs(Operation* op, int n, int slice_n, int c_slice_num, int h_slice_num) {
  int secs = 0;
  if (name() == "mlp_group" || (name() == "attention_group" && isa<tpu::MatMulOp>(op) && op == ops.back())) {
    secs = h_slice_num*c_slice_num*align(n, slice_n)/slice_n;
  } else {
    secs = c_slice_num*align(n, slice_n)/slice_n;
  }
  return secs;
}

bool speical_layer_group_base::CalcMatMulGroupTpNum(ilp_LgInfo &lg_info, Operation*& failed_op, int64_t core_num) {
  int64_t in_n, in_c, in_d, in_h, in_w, out_n, out_c, out_d, out_h, out_w, c_slice_num = 1, h_slice_num = 1;
  group_type_t type = lg_info._lgInfo.type;
  lg_info.p_special_grp->get_batch_size(lg_info.shape_secs);
  int min_test_slice_n = ceiling_func(lg_info.shape_secs.n, core_num);
  int batch_size = lg_info.shape_secs.n;

  std::vector<Operation*> tmp_ops, tmp_ops2;
  for (auto op: lg_info._lgInfo.group_ops) {
    if (isa<tpu::MatMulOp>(op)) {
      tmp_ops.push_back(op);
    } else {
      tmp_ops2.push_back(op);
    }
  }
  tmp_ops.insert(tmp_ops.end(), tmp_ops2.begin(), tmp_ops2.end());
  for (auto op: tmp_ops) {
    auto ins = get_input_values(op);
    auto outs = get_output_values(op);
    int try_c_slice_num = c_slice_num, try_h_slice_num = h_slice_num;
    int h_slice_num_ok = h_slice_num, c_slice_num_ok = c_slice_num;
    int test_slice_n = min_test_slice_n, slice_n_ok = min_test_slice_n;
    llvm::errs() << "CalcMatMulGroupTpNum for op:" <<module::getName(op).str()<<'\n';
    int secs = get_secs(op, batch_size, test_slice_n, try_c_slice_num, try_h_slice_num);
    int old_target_secs = align(secs, core_num);
    bool init_secs_is_ok =  old_target_secs == secs;
    do {
      module::getNCDHW(ins[0], in_n, in_c, in_d, in_h, in_w, type);
      llvm::errs() << "in0_n:" <<in_n<< ", in_c:" <<in_c << ", in_h:" <<in_h<< ", in_w:" <<in_w<<'\n';
      int64_t in0_lmem_bytes = 0, in1_lmem_bytes = 0, out0_lmem_bytes = 0;
      in_c = align(in_c, try_c_slice_num)/try_c_slice_num;
      if (name() == "mlp_group") {
        if (!isa<tpu::MatMulOp>(op) || op != lg_info._lgInfo.group_ops[0]) {
          in_h = align(in_h, try_h_slice_num)/try_h_slice_num;
        }
      }
      llvm::errs() <<"new in_c:" <<in_c << ", in_h:" <<in_h<<'\n';
      in0_lmem_bytes = align_64(Arch::get_tensor_lmem_bytes(ins[0], test_slice_n, in_c, in_d, in_h, in_w));

      module::getNCDHW(outs[0], out_n, out_c, out_d, out_h, out_w, type);
      llvm::errs() << "out0_n:" <<out_n<< ", out_c:" <<out_c << ", out_h:" <<out_h<< ", out_w:" <<out_w<<'\n';
      out_c = align(out_c, try_c_slice_num)/try_c_slice_num;
      if (name() == "mlp_group") {
        if (!isa<tpu::MatMulOp>(op) || op == lg_info._lgInfo.group_ops[0]) {
          out_h = align(out_h, try_h_slice_num)/try_h_slice_num;
        }
      } else if (name() == "attention_group") {
        if (isa<tpu::MatMulOp>(op) && op == ops.back()) {
          out_h = align(out_h, try_h_slice_num)/try_h_slice_num;
        }
      }
      llvm::errs() <<"new out_c:" <<out_c << ", out_h:" <<out_h<<'\n';
      out0_lmem_bytes = align_64(Arch::get_tensor_lmem_bytes(outs[0], test_slice_n, out_c, out_d, out_h, out_w));
      if (name() == "attention_group") {
        if (isa<tpu::MatMulOp>(op) && try_h_slice_num > 1 && op == ops.back()) {
          out0_lmem_bytes *= 2; //第2个matmul输入和输出占用2倍内存是考虑上一个时隙的store和下一个时隙的load分别要占一个
        }
    }

      auto lg_op = cast<LocalGenInterface>(op);
      int64_t buffer_size = lg_op.getBufferSize(in0_lmem_bytes, out0_lmem_bytes, test_slice_n, in_c,
                                        in_h, in_d, in_w, test_slice_n, out_c, out_h, out_d, out_w, type);
      if (ins.size() > 1) {
        module::getNCDHW(ins[1], in_n, in_c, in_d, in_h, in_w, type);
        llvm::errs() << "in1_n:" <<in_n<< ", in_c:" <<in_c << ", in_h:" <<in_h<< ", in_w:" <<in_w<<'\n';
        if (name() == "mlp_group") {
          if (isa<tpu::MatMulOp>(op)) {
            if (op == lg_info._lgInfo.group_ops[0]) {
              if (dyn_cast<tpu::MatMulOp>(op).getRightTranspose()) {
                in_c = align(in_c, try_h_slice_num)/try_h_slice_num;
              } else {
                in_h = align(in_h, try_h_slice_num)/try_h_slice_num;
              }
            } else {
              if (dyn_cast<tpu::MatMulOp>(op).getRightTranspose()) {
                in_h = align(in_h, try_h_slice_num)/try_h_slice_num;
              } else {
                in_c = align(in_c, try_h_slice_num)/try_h_slice_num; //注意第2个matml的右矩阵切分的行数为try_h_slice_num
              }
            }
          } else {
            in_c = align(in_c, try_c_slice_num)/try_c_slice_num;
            in_h = align(in_h, try_h_slice_num)/try_h_slice_num;
          }
        } else if (name() == "attention_group") {
          if (isa<tpu::MatMulOp>(op) && op == ops.back()) {
            in_h = align(in_h, try_h_slice_num)/try_h_slice_num;
          }
        }
        llvm::errs() <<"new in1_c:" <<in_c << ", in_h:" <<in_h<<'\n';
        in1_lmem_bytes = align_64(Arch::get_tensor_lmem_bytes(ins[1], test_slice_n, in_c, in_d, in_h, in_w));
        if (name() == "attention_group") {
        if (isa<tpu::MatMulOp>(op) && try_h_slice_num > 1 && op == ops.back()) {
            in1_lmem_bytes *= 2; //
          }
      }
      }
      bool inc_c_slice = true;
      if (isa<tpu::MatMulOp>(op)) {
      if (name() != "attention_group" || op == ops.back()) {
          inc_c_slice = in0_lmem_bytes >= in1_lmem_bytes; //相等时优先切c
      }
    }

      int total = buffer_size + in0_lmem_bytes + in1_lmem_bytes + out0_lmem_bytes;
      bool mem_enough = total <= Arch::LMEM_BYTES;
      llvm::errs() << "in0_lmem_bytes:" <<in0_lmem_bytes<< ", out0_lmem_bytes:" <<out0_lmem_bytes
                    << ", in1_lmem_bytes:" <<in1_lmem_bytes<< ", buffer_size:" <<buffer_size<< ", total:" <<total
                    << ", inc_c_slice:" <<inc_c_slice<< ", old_target_secs:" <<old_target_secs
                    << ", mem_enough:" <<mem_enough<<'\n';
      if (mem_enough) {
        if (init_secs_is_ok) {
          llvm::errs() << "init_secs_is_ok\n";
          break;
        }
        if(!inc_slice_num(test_slice_n, try_c_slice_num, try_h_slice_num, lg_info.shape_secs.c,
                          lg_info.shape_secs.h, inc_c_slice)) {
          failed_op = op;
          return false;
        }
        secs = get_secs(op, batch_size, test_slice_n, try_c_slice_num, try_h_slice_num);
        if (secs > old_target_secs) {
          llvm::errs() << "new secs("<<secs<<") >= old_target_secs, break\n";
          break;
        }
      } else {
        if(!inc_slice_num(test_slice_n, try_c_slice_num, try_h_slice_num, lg_info.shape_secs.c,
                          lg_info.shape_secs.h, inc_c_slice)) {
          failed_op = op;
          return false;
        }
        secs = get_secs(op, batch_size, test_slice_n, try_c_slice_num, try_h_slice_num);
        old_target_secs = align(secs, core_num);
        init_secs_is_ok =  old_target_secs == secs;
      }
      slice_n_ok = test_slice_n;
      c_slice_num_ok = try_c_slice_num;
      h_slice_num_ok = try_h_slice_num;
    } while(true);

    llvm::errs() << "c_slice_num_ok:" <<c_slice_num_ok<< ", h_slice_num_ok:" <<h_slice_num_ok<< ", slice_n_ok:" <<slice_n_ok<<'\n';
    if (c_slice_num_ok > c_slice_num) {
      c_slice_num = c_slice_num_ok;
    }
    if (h_slice_num_ok > h_slice_num) {
      h_slice_num = h_slice_num_ok;
    }
    if (min_test_slice_n > slice_n_ok) {
      min_test_slice_n = slice_n_ok;
    }
  }

  llvm::errs() << "fill_slice_info start\n";
  lg_info.shape_secs.n_slice_num = align(lg_info.shape_secs.n, min_test_slice_n)/min_test_slice_n;
  lg_info.shape_secs.c_slice_num = c_slice_num;
  lg_info.shape_secs.h_slice_num = h_slice_num;
  fill_slice_info(lg_info);
  if (name() == "mlp_group") {
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
    int load_cycle = cycle_calculator_->getGdmaCycle(in, info, lg_info._lgInfo.type);
    auto res = tmp_ops[0]->getResult(0);
    if (lg_info.tensor_infos.find(res) != lg_info.tensor_infos.end()) {
      info = lg_info.tensor_infos[res];
    }
    info.mode2 = TIMESTEP2_STORE;
    int store_cycle = cycle_calculator_->getGdmaCycle(res, info, lg_info._lgInfo.type);
    int bdc_cycle = cycle_calculator_->getLocalLayerCycle(tmp_ops[0], lg_info.tensor_infos,
                                        lg_info._lgInfo.type, true);
    auto diff = std::abs(bdc_cycle - store_cycle - load_cycle);
    llvm::errs()<<"h_slice_num:" <<lg_info.shape_secs.h_slice_num
                << ", load_cycle:" <<load_cycle<< ", store_cycle:" <<store_cycle<< ", bdc_cycle:" <<bdc_cycle
                << ", diff:" <<diff<<'\n';
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
  } while(true);
  std::sort(vec_hslice_and_diff_cycle.begin(), vec_hslice_and_diff_cycle.end(),pair_int_Sort_by_int);
  lg_info.shape_secs.h_slice_num = vec_hslice_and_diff_cycle[0].first;
  fill_slice_info(lg_info);
  llvm::errs()<<"find best h_slice_num:" <<lg_info.shape_secs.h_slice_num
              << ", diff:" <<vec_hslice_and_diff_cycle[0].second<<'\n';
  llvm::errs() << "n:" <<lg_info.shape_secs.n<< ", c:" <<lg_info.shape_secs.c<< ", h:" <<lg_info.shape_secs.h
              << ", n_slice_num:" <<lg_info.shape_secs.n_slice_num
              << ", c_slice_num:" <<c_slice_num<< ", h_slice_num:" <<h_slice_num<<'\n';
  return true;
}

static bool isElementwiseOp(Operation *op) {
  if (isa<tpu::ReshapeOp, tpu::ActiveOp, tpu::CastOp, tpu::MulConstOp, tpu::MulOp, tpu::AddOp, tpu::SubOp, tpu::DivOp>(op)) {
    return true;
  }
  return false;
}

static void CollectElementwiseOpAroundMatmul(Operation *mmOp, const std::vector<Operation*> &group_ops,
                                             std::vector<Operation*>& result_ops, int forward_search = 0) {
  result_ops.push_back(mmOp);
  if (forward_search == 0 || forward_search == 1) {
    for (auto v : mmOp->getOperands()) {
      auto pre_op = v.getDefiningOp();
      if (pre_op && !isa<top::NoneOp, top::WeightOp, top::InputOp>(pre_op)) {
        if (isElementwiseOp(pre_op) && std::find(group_ops.begin(), group_ops.end(), pre_op) != group_ops.end()) {
          CollectElementwiseOpAroundMatmul(pre_op, group_ops, result_ops, 1);
        }
      }
    }
  }

  if (forward_search == 0 || forward_search == 2) {
    for (auto user : mmOp->getUsers()) {
      if (isElementwiseOp(user) && std::find(group_ops.begin(), group_ops.end(), user) != group_ops.end()) {
        CollectElementwiseOpAroundMatmul(user, group_ops, result_ops, 2);
      }
    }
  }
}


class single_matmul_group : public speical_layer_group_base {
public:
  virtual bool pattern_match_and_parser(Operation* start_op, std::vector<Operation*>& subnet_ops, std::vector<Operation*>& accessed_ops) override {
    if (isa<tpu::MatMulOp>(start_op)) {
      CollectElementwiseOpAroundMatmul(start_op, subnet_ops, ops);
      auto ops_reorder = sortOpsByOtherOpsOrder(subnet_ops, ops);
      ops.assign(ops_reorder.begin(), ops_reorder.end());
    }
    return false;
  }

  virtual std::string name() override { return "single_matmul_group"; }
  virtual std::string brief() override {
    return "mlp in transformer block";
  }
  virtual bool convert_to_other_type(const std::vector<Operation*>& sub_ops, std::shared_ptr<speical_layer_group_base>& p_special_grp) override {
    for (auto op: sub_ops) {
      if (isa<tpu::MatMulOp>(op)) {
        return true;
      }
    }
    return false;
  }
};


class mlp_group : public speical_layer_group_base {
public:
  virtual bool pattern_match_and_parser(Operation* start_op, std::vector<Operation*>& subnet_ops,
                                        std::vector<Operation*>& accessed_ops) override {
    Operation* next_matmul_op = nullptr;
    if (search_two_mmOp(start_op, next_matmul_op, subnet_ops, accessed_ops)) {
      auto mmOp = dyn_cast<tpu::MatMulOp>(next_matmul_op);
      if (dyn_cast<tpu::MatMulOp>(start_op).getHdimIsBatch() || mmOp.getHdimIsBatch()) {
        return false; //不包含带HdimIsBatch的matmul
      }
      map_value_to_cut_dims[mmOp.getRight()] = {0,3,2,1,4};
      std::vector<Operation *> del_ops;
      for (auto op: ops) {
        if (isa<tpu::MatMulOp>(op)) {
          auto next_op = *(op->getUsers().begin());
          if (isa<tpu::ReshapeOp>(next_op) && std::find(ops.begin(), ops.end(), next_op) != ops.end()) {
            assert((op->getResult(0).hasOneUse()));
            bool has_out_user = false;
            for (auto user: next_op->getUsers()) {
              if (find(ops.begin(), ops.end(), user) == ops.end()) {
                has_out_user = true;
                break;
              }
            }
            if (!has_out_user) {
              del_ops.push_back(next_op);
            }
            op->getResult(0).setType(next_op->getResult(0).getType());
            next_op->getResult(0).replaceUsesWithIf(op->getResult(0), [&](OpOperand &operand) {
              Operation *user = operand.getOwner();
              return find(ops.begin(), ops.end(), user) != ops.end();
            });
          } else {
            if (op == ops[0]) {
              break;
            }
          }

          auto inOp = dyn_cast<tpu::MatMulOp>(op).getInput().getDefiningOp();
          if (inOp && isa<tpu::ReshapeOp>(inOp)) {
            bool has_out_user = false;
            for (auto user: inOp->getUsers()) {
              if (find(ops.begin(), ops.end(), user) == ops.end()) {
                has_out_user = true;
                break;
              }
            }
            if (!has_out_user) {
              del_ops.push_back(inOp);
            }
            auto oldType = inOp->getOperand(0).getType();
            inOp->getResult(0).replaceUsesWithIf(inOp->getOperand(0), [&](OpOperand &operand) {
              Operation *user = operand.getOwner();
              return find(ops.begin(), ops.end(), user) != ops.end();
            });
            op->getOperand(0).setType(oldType);
          }
        }
      }
      for (auto del_op: del_ops) {
        del_op->erase();
        ops.erase(std::remove(ops.begin(), ops.end(), del_op), ops.end());
        subnet_ops.erase(std::remove(subnet_ops.begin(), subnet_ops.end(), del_op), subnet_ops.end());
      }

      for (auto op: ops) {
        if (isa<tpu::ConcatOp>(op)) {
          auto in_shape = module::getShapeVec(op->getOperand(0));
          if (in_shape.size() == dyn_cast<tpu::ConcatOp>(op).getAxis() + 1) {
            col_cut = false;
          }
        }
      }
      return ops.size() > 1;
    }
    return false;
  }

  virtual std::string name() override { return "mlp_group"; }
  virtual std::string brief() override {
    return "mlp in transformer block";
  }

  virtual bool convert_to_other_type(const std::vector<Operation*>& sub_ops,
                                     std::shared_ptr<speical_layer_group_base>& p_special_grp) override {
    if (isa<tpu::MatMulOp>(sub_ops[0])) {
      ops.assign(sub_ops.begin(), sub_ops.end());
    } else {
      p_special_grp = std::make_shared<single_matmul_group>();
      p_special_grp->ops.assign(sub_ops.begin(), sub_ops.end());
    }
    return true;
  }
};

class attention_group : public speical_layer_group_base {
public:
  virtual bool pattern_match_and_parser(Operation* start_op, std::vector<Operation*>& subnet_ops, std::vector<Operation*>& accessed_ops) override {
    Operation* next_matmul_op = nullptr;
    if (search_two_mmOp(start_op, next_matmul_op, subnet_ops, accessed_ops)) {
      return find_softmax;
    }
    return false;
  }
  virtual std::string name() override { return "attention_group"; }
  virtual std::string brief() override {
    return "attention in transformer block";
  }

  virtual bool convert_to_other_type(const std::vector<Operation*>& sub_ops, std::shared_ptr<speical_layer_group_base>& p_special_grp) override {
    if (isa<tpu::MatMulOp>(sub_ops.back())) {
      ops.assign(sub_ops.begin(), sub_ops.end());
      return true;
    } else {
      // p_special_grp = std::make_shared<single_matmul_group>();
      // p_special_grp->ops.assign(sub_ops.begin(), sub_ops.end());
      return false;
    }
  }

};

std::vector<std::shared_ptr<speical_layer_group_base>>
GroupOps::findSpecialGroup(std::vector<Operation*>& subnet_ops) {
  std::vector<Operation*> accessed_ops;
  std::vector<std::shared_ptr<speical_layer_group_base>> vec_group_ptr;
  if (module::isDebugCmdEnable("disable_findSpecialGroup")) {
    return std::move(vec_group_ptr);
  }
  for (auto op: subnet_ops) {
    if (module::isDebugCmdEnable("user_defined_global_op")) {
      if (module::isDebugCmdEnable("-" + module::getName(op).str())) {
        accessed_ops.push_back(op);
      }
    }
  }
  for (auto op: subnet_ops) {
    if (std::find(accessed_ops.begin(), accessed_ops.end(), op) == accessed_ops.end()) {
      auto attention_grp_ptr = std::make_shared<attention_group>();
      if (attention_grp_ptr->pattern_match_and_parser(op, subnet_ops, accessed_ops)) {//first match attention block
        vec_group_ptr.push_back(attention_grp_ptr);
        accessed_ops.insert(accessed_ops.end(), attention_grp_ptr->ops.begin(), attention_grp_ptr->ops.end());
      } else {
        continue;
      }
      llvm::errs()<<"find a "<<vec_group_ptr.back()->name()<<":\n";
      for (auto it: vec_group_ptr.back()->ops) {
        llvm::errs()<<show_op_info(it)<<"\n";
      }
    }
  }

  for (auto op: subnet_ops) {
    if (std::find(accessed_ops.begin(), accessed_ops.end(), op) == accessed_ops.end()) {
      auto mlp_group_ptr = std::make_shared<mlp_group>();
      if (mlp_group_ptr->pattern_match_and_parser(op, subnet_ops, accessed_ops)) {
        vec_group_ptr.push_back(mlp_group_ptr);
        accessed_ops.insert(accessed_ops.end(), mlp_group_ptr->ops.begin(), mlp_group_ptr->ops.end());
      } else {
        continue;
      }
      llvm::errs()<<"find a "<<vec_group_ptr.back()->name()<<":\n";
      for (auto it: vec_group_ptr.back()->ops) {
        llvm::errs()<<show_op_info(it)<<"\n";
      }
    }
  }

  // for (auto op: subnet_ops) {
  //   if (std::find(accessed_ops.begin(), accessed_ops.end(), op) == accessed_ops.end()) {
  //     auto mlp_group_ptr = std::make_shared<single_matmul_group>();
  //     if (mlp_group_ptr->pattern_match_and_parser(op, subnet_ops, accessed_ops)) {
  //       vec_group_ptr.push_back(mlp_group_ptr);
  //       accessed_ops.insert(accessed_ops.end(), mlp_group_ptr->ops.begin(), mlp_group_ptr->ops.end());
  //     } else {
  //       continue;
  //     }
  //     llvm::errs()<<"find a "<<vec_group_ptr.back()->name()<<":\n";
  //     for (auto it: vec_group_ptr.back()->ops) {
  //       llvm::errs()<<show_op_info(it)<<"\n";
  //     }
  //   }
  // }
  llvm::errs()<<"vec_group_ptr.size:"<<vec_group_ptr.size()<<"\n";
  return std::move(vec_group_ptr);
}

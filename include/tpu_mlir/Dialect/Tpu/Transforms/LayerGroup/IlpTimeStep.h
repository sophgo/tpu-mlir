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
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupDefs.h"
#include "ortools/linear_solver/linear_solver.h"

namespace tpu_mlir {
namespace tpu {

using namespace operations_research;

typedef struct reload_mem_struct {
 int addr;
 std::vector<int> bank_id;
} reload_mem_struct;

typedef struct his_mem_struct {
 int type;
 int size;
 int slice_idx;
 Value value;
  std::vector<std::pair<int64_t, reload_mem_struct>> vec_reload_addr;
} his_mem_struct;


typedef struct mem_struct {
 int type; //0: normal tensor mem, 1:free mem , 2:reside value mem, 3:op buffer mem
 int addr;
 int size;
 int refc;
 std::vector<int> bank_id;
 int slice_idx;
 Value value;
//  std::string mnemonic_name;
} mem_struct;

typedef struct ilp_var_info {
 int ts_idx;
 int slice_idx;
//  int use_or_gen_ts;
 int store_load_mode = -1;
 MPVariable* ilp_var;
 tensor_info_t tensor_info;
} ilp_var_info;

typedef struct store_load_info_t {
  Value value;
  int ada_stay_mem_size = 0;
  int slice_idx;
  Operation* near_op = nullptr;
  std::vector<std::pair<std::string, int>> store_vars;
  std::vector<std::pair<std::string, int>> load_vars;
} store_load_info_t;

typedef struct mem_alloc_status {
 bool failed = false;
 Value load_fail_value;
} mem_alloc_status;

typedef struct reside_value_info {
 int addr;
 int size;
} reside_value_info;

typedef struct ts_move_info {
  std::string name;
  std::vector<Value> move_value;
  std::vector<int64_t> move_src_add;
  std::vector<int64_t> move_dest_add;
  std::vector<int64_t> move_size;
  std::vector<int64_t> slice_idx;
} ts_move_info;

typedef struct cons_info {
  int store_cons_idx = -1;
  int load_cons_idx = -1;
  std::vector<std::string> store_var_names;
  std::vector<std::string> load_var_names;
} cons_info;

typedef struct mem_alloc_req_info {
  int slice_idx;
  int size;
  std::string name;
  Value value;
  bool isBuffer = false;
  mem_alloc_req_info()
      : slice_idx(0), size(0) {}
} mem_alloc_req_info;

typedef struct constraint_info {
  double lb;
  double ub;
  MPConstraint* cons_var;
  std::string info_for_tips;
  std::vector<std::pair<int, MPVariable*>> coeff_var_items;
  bool operator==(const constraint_info &other) const {
    return this == &other;
  }
} constraint_info;

typedef struct ts_mem_cons {
  int cons_idx;
  Operation* op;
} ts_mem_cons;

class ILPTimeStep;
class lmem_alloc {
public:
  lmem_alloc(std::map<std::string, std::vector<std::string>>& banked_tensors, ILPTimeStep* pILPTimeStep, int ts_count);
  virtual ~lmem_alloc();

  std::shared_ptr<std::vector<std::pair<std::string, mem_struct>>> show_mem(int& total_free_size, int& max_free_mem_idx, int& max_free_mem_size);
  bool alloc(int ts_idx, int slice_idx, const std::string& name, Value value, int size, bool isBuffer = false);
  bool alloc2(int slice_idx, const std::string& name, Value value, int addr, int size);
  bool free(const std::string& name, std::vector<std::pair<int,int>>* vec_pre_ts_free_mem = nullptr);
  bool get_mem_struct(const std::string& key, mem_struct& mem_s);

// private:
  std::vector<int> get_bank(const std::string& name);
  bool _alloc(int slice_idx, const std::string& name, Value value, int size, std::vector<int>& ret_bank_id,
              int& free_addr, int& confict_size, bool force_not_care_bank = false,
              bool isBuffer = false, bool for_move = false);
  bool alloc_multi(int ts_idx, std::vector<mem_alloc_req_info>& vec_mem_req, bool sort_by_size);

// private:
  int total_size;
  int m_ts_count;
  // int bank_size;
  bool* lmem_buf;
  std::map<std::string, mem_struct> mem_dict;
  std::map<std::string, his_mem_struct> vec_mem_alloc_his;
  int bank_num[16];
  int bank_area_start_addr[17];
  std::map<std::string, std::vector<std::string>>& banked_tensors_;
  ILPTimeStep* m_pILPTimeStep;
  bool rehearsal = false;
};

class l2mem_alloc {
public:
  l2mem_alloc();
  virtual ~l2mem_alloc();

  bool alloc(int slice_idx, const std::string& name, Value value, int size);
  bool free(int slice_idx, const std::string& name);
  void clear();

// private:
  int total_size;
  int m_ts_count;
  bool* lmem_buf;
  std::map<std::string, mem_struct> mem_dict;
  std::map<std::string, his_mem_struct> vec_mem_alloc_his;
};

using lmem_alloc_Ptr = std::shared_ptr<lmem_alloc>;
using l2mem_alloc_Ptr = std::shared_ptr<l2mem_alloc>;

struct value_compare2 {
  bool operator()(const std::pair<Value, int>& v0, const std::pair<Value, int>& v1) const {
    if (v0.first.getImpl() != v1.first.getImpl()) {
      return v0.first.getImpl() < v1.first.getImpl();
    }
    return v0.second < v1.second;
  }
};

class dot_graph;
class ILPTimeStep {
public:
  ILPTimeStep(const LgInfo& group_info, std::shared_ptr<dot_graph> tmp_dot_graph_log, int sec_per_core = 1);
  virtual ~ILPTimeStep();

  // static ILPTimeStep& combine_mult_ilp_timestep(
  //   const std::vector<ILPTimeStep&> other_ilp_timesteps) {};

  void addBinaryVar(int ts_idx, int slice_idx, int mode, std::string varName, Value value, tensor_info_t& info, int64_t lmem_bytes);
  void addTimestepGdmaCycle(int ts_idx, int cycle, std::string varName);
  void addTimestepMemUse(int ts_idx, int mem_size, std::vector<std::string>& varNames);
  void addNewOutIntoReturnOp(std::vector<std::string> var_names, Value value);
  void addRowConstraint(int ts_idx, Value value, std::vector<std::string> var_names, bool store, bool is_weight);
  void setVarExpectValue(std::string var_name, int expect_value);
  bool run(Operation*& fail_op);
  bool mem_alloc(mem_alloc_status& alloc_status, std::vector<std::pair<Value, int64_t>>& value_size,
                TensorInfo& tensor_infos, Operation*& fail_op);
  void add_tpu_ts_field(int ts_idx, Operation* op);
  void add_gdma_ts_field(int ts_idx, const GdmaElt &field);
  // void timestep_assignment_by_ilp(BasicTimeStep *time_step, TensorInfo &tensor_infos);
  int get_tensor_load_pos(const tensor_info_t& tensor_info, std::string& var_name);
  void get_group_cycle_info(int& total_cycle, int& total_diff,
    std::vector<std::pair<int, std::vector<Operation*>>>& ts_cycle_diff);

  bool merge_small_cycle_op(TensorInfo& tensor_infos, std::shared_ptr<dot_graph> dot_graph_log);
  bool prepare(TensorInfo& tensor_infos);
  void addTensorSize(Value value, int slice_idx, int lmem_size);
  void addTensorCycle(Value value, int slice_idx, int cycle);
  void addOpInfo(int ts_idx, Operation* op, int buffer_size, int mem_size_for_load, int bdc_cycle);
  void addValueInfo(int slice_idx, Value value, std::string varName);
  MPVariable* getMPVarByName(std::string varName);
  void addSliceNcdhwSteps(int core_id, std::vector<int64_t> ncdhw);
  void resideOpInValue(Operation* op, Value value);
  void showTimeStepInfo(int debug_cmd = 0);
  void showAllConstraint();
  void showRunInfo();
  void showConstraintInfo(constraint_info& cons_info, std::map<MPVariable*, std::string>& only_one_var_warning);
  int addConstraint(double lb, double ub, std::vector<std::pair<int, MPVariable*>> coeff_var_items,
                    std::string info_for_tips = "", bool test = true);
// protected:
  LgInfo _group_info;
  std::unique_ptr<MPSolver> solver;
  std::vector<TimestepRow2> timestep_table_;
  std::vector<TimestepRow2> timestep_table_new;
  std::map<int64_t, std::vector<ts_move_info>> inserted_timestep_table_;
  std::map<std::string, ilp_var_info> mapILPVarInfo;
  std::vector<std::vector<std::pair<int, std::string>>> cycle_contrains;
  std::vector<std::vector<std::pair<int, std::string>>> mem_contrains;
  std::vector<std::vector<std::pair<int, std::string>>> cycle_contrains_new;
  std::vector<std::vector<std::pair<int, std::string>>> mem_contrains_new;
  std::vector<ts_mem_cons> ts_mem_contrains;
  MPObjective* objective;
  lmem_alloc_Ptr lmem_alloc_ptr;
  std::vector<l2m_value_info> vec_l2m_value_info; //Value + ts_idx > 加载ts号
  std::map<Value, std::map<int, std::vector<std::string>>, value_compare> mapValueInfo;
  std::map<Value, std::map<int, cons_info>, value_compare> mapConsInfo;
  std::vector<constraint_info> vec_constraints;

  std::vector<int64_t> core_ids;
  std::map<int, std::vector<std::vector<int64_t>>> ncdhw_steps;
  std::map<Operation*, std::vector<Value>> reside_in_tensor; //分叉的输出因为远处的user靠得太近，不必要store/load，故直接驻留
  std::map<Value, std::vector<std::string>, value_compare> values_need_store_to_grpout;
  std::map<Value, reside_value_info, value_compare> map_reside_value_info;
  std::map<Value, int, value_compare> mapValueUserCount;
  std::vector<Value> vecFreeByStoreVar;
  std::map<std::pair<Value, int>, int, value_compare2> load_tensor_cycles;
  std::map<std::pair<Value, int>, int, value_compare2> dam_tensor_size;
  std::map<int, int> map_merge_start_to_merge_len;
  std::shared_ptr<dot_graph> dot_graph_log;
// private:
  int ts_count;
  int slice_num;
  int prepare_offset;
  int m_sec_per_core;
  bool ada_load_store = true;
  int m_constraint_idx = 0;
  bool detail_log = false;
};

using ILPTimeStepPtr = std::shared_ptr<ILPTimeStep>;

} // namespace tpu
} // namespace tpu_mlir


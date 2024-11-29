//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Support/Module.h"
#include "llvm/Support/FormatVariadic.h"
#include <list>
#include <map>
#include <set>
#include <vector>

// using namespace tpu_mlir::helper;
namespace tpu_mlir {
namespace tpu {

typedef struct {
  int64_t nstep;
  int64_t cstep;
  int64_t hstep;
  int64_t dstep;
  int64_t wstep;
} tensor_step_t;

typedef enum ld_st_type {
  TIMESTEP_LOAD = 0,    // load from gmem to lmem
  TIMESTEP_STORE = 1,   // store from lmem to gmem
  TIMESTEP_MOVE = 2,    // move between global mem
  TIMESTEP_LD_G2L2 = 3, // load from gmem to l2mem
  TIMESTEP_LDST_UNKNOWN
} TIMESTEP_LD_ST;

typedef enum ld_st_type2 {
  TIMESTEP2_LOAD = 1,    // load from gmem to lmem
  TIMESTEP2_STORE = 2,   // store from lmem to gmem
  TIMESTEP2_MOVE = 4,    // move between global mem
  TIMESTEP2_LD_G2L2 = 8, // load from gmem to l2mem
  TIMESTEP2_STORE_AND_LOAD =
      16, // first store, and skip some timesteps, then load
  // TIMESTEP2_STORE_ONLY_FREE = 32, //only free
  TIMESTEP2_MOVE_BTW_LMEM = 64, // move between lmem mem
  TIMESTEP2_ONLY_RESIDE = 128,
  TIMESTEP2_LDST_UNKNOWN
} TIMESTEP_LD_ST2;

inline bool is_timestep_load(TIMESTEP_LD_ST type) {
  return (type == TIMESTEP_LOAD || type == TIMESTEP_LD_G2L2);
}

typedef enum {
  LMEM_WEIGHT,
  LMEM_ACTIVATION,
  LMEM_OPERATION,
  LMEM_ANY,
} lmem_type_t;

typedef std::pair<int64_t, int64_t> slice_pair_t; // idx and slice
struct slice_info_t {
  std::vector<slice_pair_t> h; // h_idx and h_slice
  std::vector<slice_pair_t> n; // n_idx and n_slice
  std::vector<slice_pair_t> w; // w_idx and w_slice
  std::vector<slice_pair_t> d; // d_idx and d_slice
  std::vector<slice_pair_t> c; // c_idx and c_slice
};

typedef struct mem_buffer_key {
  lmem_type_t type;
  Value value;
  Operation *op;
  int64_t conflict;
  bool operator<(const mem_buffer_key &other) const {
    if (type < other.type) {
      return true;
    } else if (type == other.type) {
      if (type == LMEM_OPERATION) {
        return op < other.op;
      } else {
        return value.getImpl() < other.value.getImpl();
      }
    }
    return false;
  }
} mem_buffer_key_t;

typedef struct mem_buffer_value {
  int64_t start_ts;
  int64_t end_ts;
  int64_t addr;
  int64_t size;
  int64_t align_bytes;
} mem_buffer_value_t;

struct value_compare {
  bool operator()(Value v0, Value v1) const {
    if (v0.getImpl() < v1.getImpl()) {
      return true;
    }
    return false;
  }
};

struct ptr_compare {
  bool operator()(Operation *v0, Operation *v1) const {
    if ((int64_t)v0 < (int64_t)v1) {
      return true;
    }
    return false;
  }
};

struct tensor_info_t {
  TIMESTEP_LD_ST mode;
  int64_t mode2;
  std::map<Operation *, slice_info_t, ptr_compare> slice_infos;
  slice_info_t slice_info;
  int64_t stage;
  int64_t use_3ic_opt;
  bool eu_align;
  bool need_bcast;
  bool hold_in_lmem;

  // init
  tensor_info_t()
      : mode(TIMESTEP_LOAD), stage(0), use_3ic_opt(0), eu_align(false),
        mode2(0), need_bcast(false) {}
  tensor_info_t(TIMESTEP_LD_ST mode)
      : mode(mode), stage(0), use_3ic_opt(0), eu_align(false), mode2(0),
        need_bcast(false) {}
  tensor_info_t(slice_info_t slice_info)
      : slice_info(slice_info), mode(TIMESTEP_LDST_UNKNOWN), stage(0),
        use_3ic_opt(0), mode2(0), eu_align(false), need_bcast(false) {}
  tensor_info_t(Operation *next_op, slice_info_t slice_info)
      : slice_info(slice_info), mode(TIMESTEP_LDST_UNKNOWN), stage(0),
        use_3ic_opt(0), mode2(0), eu_align(false), need_bcast(false) {
    slice_infos[next_op] = slice_info;
  }
  void add_slice_info(Operation *next_op, slice_info_t slice_info) {
    slice_infos[next_op] = slice_info;
  }
};

using ValueSet = std::set<Value, value_compare>;
using ValueIntMap = std::map<Value, int64_t, value_compare>;
using IntValueIntMap = std::map<int64_t, ValueIntMap>;
using TensorInfo = std::map<Value, tensor_info_t, value_compare>;
using MemBuff = std::map<mem_buffer_key_t, mem_buffer_value_t>;
using MemBuffElt = std::pair<mem_buffer_key_t, mem_buffer_value_t>;
using TpuTsField = std::vector<Operation *>;
using GdmaElt = std::pair<Value, tensor_info_t>;
using GdmaTsField = std::vector<GdmaElt>;
using MemBlock = std::pair<int64_t, int64_t>; // <addr, size>

typedef struct {
  TpuTsField tpu0_ts_field;
  GdmaTsField gdma0_ts_field;
} TimestepRow;

typedef struct ts_var_t {
  std::string varName;
  int var_value;
  int slice_idx;

  Value value;
  int lmem_bytes;
  tensor_info_t info;
  ts_var_t() : var_value(0), lmem_bytes(0), slice_idx(-1) {}
} ts_var_t;

struct op_related_info_t {
  Operation *op;
  int slice_idx;
  int buffer_size;
  int bdc_cycle;
  int mem_size_for_load;
  std::map<Value, std::vector<std::string>, value_compare>
      vars_need_load_to_l2m;
  std::map<Value, int, value_compare> tensor_size;
  std::map<Value, int, value_compare> load_tensor_cycles;
  // std::vector<ts_var_t>  ada_var_for_free_mem;
  // //在本op执行后可以释放的自动驻留输入tensor
  op_related_info_t() : op(nullptr) {}
};

typedef struct TimestepRow2 {
  // int cycle_diff;
  bool can_merge = false;
  std::vector<ts_var_t> vec_ts_var;
  std::vector<op_related_info_t> vec_op_infos;
} TimestepRow2;

typedef struct op_var_pos_info {
  std::pair<int, int> key;
  int ts_id;
  int start_ts;
  int end_ts;
  op_var_pos_info() : ts_id(-1), key(std::make_pair(-1, -1)) {}
} op_var_pos_info;

typedef struct l2m_value_info {
  Value value;
  int slice_idx;
  int size;
  int free_ts;
  int load_ts;
  bool valid;
  l2m_value_info()
      : slice_idx(0), size(0), free_ts(0), load_ts(0), valid(true) {}
} l2m_value_info;

typedef struct shape_secs {
  int64_t nsecs;
  int64_t hsecs;
  int64_t dsecs;
  int64_t wsecs;
  int64_t csecs;

  int64_t n;
  int64_t c;
  int64_t h;
  int64_t n_slice_num;
  int64_t c_slice_num;
  int64_t h_slice_num;

  shape_secs() {
    nsecs = hsecs = dsecs = wsecs = csecs = 1;
    c_slice_num = h_slice_num = n_slice_num = n = c = h = -1;
  }
  int64_t get_sec_num(bool only_nc = false) {
    if (c_slice_num > 0) {
      if (only_nc) {
        return n_slice_num * c_slice_num;
      } else {
        return n_slice_num * c_slice_num * h_slice_num;
      }
    } else {
      return nsecs * hsecs * dsecs * wsecs * csecs;
    }
  }

  std::string info() {
    return llvm::formatv("nsecs:{0}, csecs:{1}, dsecs:{2}, hsecs:{3}, "
                         "wsecs:{4}, n:{5}, c:{6}, h:{7}, n_slice_num:{8}, "
                         "c_slice_num:{9}, h_slice_num:{10}",
                         nsecs, csecs, dsecs, hsecs, wsecs, n, c, h,
                         n_slice_num, c_slice_num, h_slice_num)
        .str();
  }

} shape_secs_t;

typedef enum {
  NOT_CHECK = 0,
  NOT_VALID = 1,
  VALID = 2,
} group_valid_type_t;

struct LgInfo {
  LgInfo() { this->clear(); }
  ~LgInfo() { this->clear(); }
  void clear() {
    this->group_banked_tensors.clear();
    this->group_ops.clear();
    this->group_ins.clear();
    this->group_outs.clear();
    this->group_op_outs.clear();
    this->type = GROUP_NORMAL;
    this->base_group_idx = -1;
    this->cache_key = -1;
    this->group_cost = 0;
    this->is_valid = NOT_CHECK;
  }

  void update_group_io(int opt = 2) {
    _opt = opt;
    this->group_ins.clear();
    this->group_outs.clear();
    this->group_op_outs.clear();

    for (auto op : group_ops) {
      if (!op) {
        continue;
      }
      // update group_ins
      for (auto in : op->getOperands()) {
        auto src_op = in.getDefiningOp();
        bool value_in;
        if (src_op != nullptr) {
          if (opt == 2 || opt == 1) {
            value_in = module::isTrain()
                           ? !isa<top::NoneOp>(src_op)
                           : !isa<top::WeightOp, top::NoneOp>(src_op);
          }
          if (opt == 3) {
            value_in = !isa<top::NoneOp>(src_op);
          }
        }
        if ((src_op == nullptr ||
             (value_in && (std::find(group_ops.begin(), group_ops.end(),
                                     src_op) == group_ops.end()))) &&
            std::find(group_ins.begin(), group_ins.end(), in) ==
                group_ins.end()) {
          group_ins.push_back(in);
        }
      }
      // update group_outs
      for (auto out : op->getResults()) {
        if (module::isNone(out)) {
          continue;
        }
        group_op_outs.push_back(out);
        for (auto dst_op : out.getUsers()) {
          if (std::find(group_ops.begin(), group_ops.end(), dst_op) ==
                  group_ops.end() &&
              std::find(group_outs.begin(), group_outs.end(), out) ==
                  group_outs.end()) {
            group_outs.push_back(out);
            break;
          }
        }
      }
    }
  }

  void update_bank_info() {
    group_banked_tensors.clear();
    for (auto op : group_ops) {
      if (!op) {
        continue;
      }
      if (op->getNumOperands() > 1) {
        auto pre_op = op->getOperand(1).getDefiningOp();
        if ((pre_op == nullptr || !isa<top::NoneOp>(pre_op))) {
          auto opd0 = module::getName(op->getOperand(0)).str();
          auto opd1 = module::getName(op->getOperand(1)).str();
          group_banked_tensors[opd0].push_back(opd1);
          group_banked_tensors[opd1].push_back(opd0);
        }
      }

      for (int i = 0; i < op->getNumOperands(); i++) {
        auto pre_op = op->getOperand(i).getDefiningOp();
        if (pre_op == nullptr || !isa<top::NoneOp>(pre_op)) {
          auto opd = module::getName(op->getOperand(i)).str();
          for (int j = 0; j < op->getNumResults(); j++) {
            auto res = module::getName(op->getResult(j)).str();
            group_banked_tensors[opd].push_back(res);
          }
        }
      }

      for (int i = 0; i < op->getNumResults(); i++) {
        auto res = module::getName(op->getResult(i)).str();
        for (int j = 0; j < op->getNumOperands(); j++) {
          auto pre_op = op->getOperand(j).getDefiningOp();
          if (pre_op == nullptr || !isa<top::NoneOp>(pre_op)) {
            auto opd = module::getName(op->getOperand(j)).str();
            group_banked_tensors[res].push_back(opd);
          }
        }
      }
    }
  }
  void const dump_lginfo() const {
    llvm::dbgs() << "LgInfo Begin {"
                 << "\n";
    llvm::dbgs() << "ins"
                 << "\n";
    for (auto op : group_ins) {
      op.dump();
    }
    llvm::dbgs() << "ops"
                 << "\n";
    for (auto op : group_ops) {
      op->dump();
    }
    llvm::dbgs() << "outs"
                 << "\n";
    for (auto op : group_outs) {
      op.dump();
    }
    llvm::dbgs() << "} LgInfo End;"
                 << "\n";
  }

  // group layers
  std::vector<Operation *> group_ops; /**cached, by loc name */
  // std::vector<Operation *> edge_ops;
  // //寻找所有preOp或nextOp都在组外的op，即组的边缘op

  // in tensors
  std::vector<Value> group_ins;
  // out tensors
  std::vector<Value> group_outs;
  // all op out tensors
  std::vector<Value> group_op_outs;
  bool use_cache = false;
  shape_secs_t shape_secs; /**cached */
  // layer group type
  group_type_t type;
  int64_t group_id = 0;

  group_valid_type_t is_valid = NOT_CHECK;
  int64_t cache_key = -1;      /**indicate if lg_info is load by cached */
  int64_t base_group_idx = -1; /**cached */
  int64_t start_idx = 0;       /**cached */
  int64_t end_idx = 0;         /**cached */
  int64_t group_cost;          /**cached */
  int64_t sort_index =
      0; /**only use for loading and dumping cache, 'index' in json*/

  std::vector<int> free_cores;
  std::map<std::string, std::vector<std::string>> group_banked_tensors;
  int _opt = 2;
};

typedef enum {
  STRATEGY_NORMAL = 0,
  STRATEGY_SEARCH_CONV_CUT = 1,
  STRATEGY_SLICE_CUT_FIRST = 2,
  STRATEGY_GROUP_CUT_FIRST = 3
} solver_strategy_type_t;

struct LgPassIR;
class CycleCalculator;
class dot_graph;
class l2mem_alloc;
class ILPTimeStep;
class speical_layer_group_base;
struct ilp_LgInfo {
  solver_strategy_type_t _cur_strategy = STRATEGY_NORMAL;
  LgInfo _lgInfo;
  std::vector<Operation *> global_layers;
  std::vector<Operation *> failed_ops;
  std::vector<Operation *> backup_ops;
  bool is_fail_op_in_grp = true;

  std::vector<std::shared_ptr<ilp_LgInfo>> sub_ilp_LgInfos;
  bool group_success = false;

  // 考察是否将global conv作为分割点
  int group_cycle = 0;
  bool conv_cut_optimized = false;
  std::shared_ptr<speical_layer_group_base> p_special_grp = nullptr;
  std::map<Value, int, value_compare> value_load_to_l2m;
  std::map<Value, int, value_compare> value_store_to_l2m;

  // 二进制搜索可行分割点
  std::vector<Operation *> group_ops_all;
  std::vector<Operation *> divided_group_ops;
  std::map<Operation *, std::vector<Operation *>> map_parallel_node;
  int middle_ptr = -1;
  int last_success_middle_ptr = -1;
  int pre_middle_ptr = -1;
  int left_ptr = 0;
  int right_ptr = 0;
  static int group_count;

  shape_secs_t shape_secs;
  TensorInfo tensor_infos;
  std::vector<std::shared_ptr<ILPTimeStep>> timeStepPtrs;
  std::map<int, std::vector<l2m_value_info>> map_l2m_load;
  TensorInfo lg_tensor_infos_;
  std::shared_ptr<l2mem_alloc> l2mem_alloc = nullptr;

  ilp_LgInfo(solver_strategy_type_t cur_strategy = STRATEGY_NORMAL) {
    _cur_strategy = cur_strategy;
    _lgInfo.group_id = group_count++;
  }
  ilp_LgInfo(const ilp_LgInfo &other) {
    _lgInfo.group_id = other._lgInfo.group_id;
    _lgInfo.group_ops.assign(other._lgInfo.group_ops.begin(),
                             other._lgInfo.group_ops.end());
    _lgInfo.update_bank_info();
    _lgInfo.update_group_io();
  }

  void base_solver(LgPassIR *pass_ir,
                   std::shared_ptr<CycleCalculator> cycle_calculator_);
  std::shared_ptr<ilp_LgInfo>
  high_solver(LgPassIR *pass_ir,
              std::shared_ptr<CycleCalculator> cycle_calculator_);
  bool binary_search_group(bool move_right,
                           std::shared_ptr<dot_graph> dot_graph_log = nullptr);
  std::vector<Operation *> GetParallelNodes(Operation *op);
  void save_result(LgPassIR *pass_ir);
};

class speical_layer_group_base {
public:
  speical_layer_group_base() {}
  virtual ~speical_layer_group_base() {}

  virtual bool
  pattern_match_and_parser(Operation *start_op,
                           std::vector<Operation *> &subnet_ops,
                           std::vector<Operation *> &accessed_ops) = 0;
  virtual std::string name() = 0;
  virtual std::string brief() { return ""; }
  virtual bool convert_to_other_type(
      const std::vector<Operation *> &sub_ops,
      std::shared_ptr<speical_layer_group_base> &p_special_grp) = 0;

  bool search_two_mmOp(Operation *start_op, Operation *&next_mmOp,
                       std::vector<Operation *> &subnet_ops,
                       std::vector<Operation *> &accessed_ops);
  void get_batch_size(shape_secs_t &shape_secs);
  bool update_shape_secs_for_ilp_group(shape_secs_t &shape_secs,
                                       const shape_secs_t &max_shape_secs);
  void fill_slice_info(ilp_LgInfo &ilp_lg_info);
  bool inc_slice_num(int &test_slice_n, int &try_c_slice_num,
                     int &try_h_slice_num, int max_c_slice_num,
                     int max_h_slice_num, bool inc_c_slice = true);
  int get_secs(Operation *op, int n, int slice_n, int c_slice_num,
               int h_slice_num);
  bool CalcMatMulGroupTpNum(ilp_LgInfo &lg_info, Operation *&failed_op,
                            int64_t core_num);

  std::vector<Operation *> ops;
  bool col_cut = true;
  bool find_softmax = false;
  std::map<Value, std::vector<int>, value_compare> map_value_to_cut_dims;
};

} // namespace tpu
} // namespace tpu_mlir

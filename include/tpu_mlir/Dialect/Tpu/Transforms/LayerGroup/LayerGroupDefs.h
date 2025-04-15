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
#include "tpu_mlir/Interfaces/LocalGenInterface.h"
#include "tpu_mlir/Support/Module.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include <list>
#include <map>
#include <set>
#include <vector>

// using namespace tpu_mlir::helper;
namespace tpu_mlir {
namespace tpu {

enum class NnvlcMode { NONE = 0, WEIGHT = 1, ACTIVATION = 2, ALL = 3 };

typedef struct {
  bool dyn_compile;
  int64_t opt;
  bool group_by_cores;
  NnvlcMode nnvlc_mode;
  bool lgcache;
  int64_t num_core;
  int64_t debugger;
  std::string debugger_filename;
  bool disable_group_overlap;
} LgOptions;

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

  std::string lmem_type_str() {
    switch (type) {
    case LMEM_WEIGHT:
      return "LMEM_WEIGHT";
    case LMEM_ACTIVATION:
      return "LMEM_ACTIVATION";
    case LMEM_OPERATION:
      return "LMEM_OPERATION";
    case LMEM_ANY:
      return "LMEM_ANY";
    }
    return "LMEM_UNKNOWN";
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
        mode2(0), need_bcast(false), hold_in_lmem(true) {}
  tensor_info_t(TIMESTEP_LD_ST mode)
      : mode(mode), stage(0), use_3ic_opt(0), eu_align(false), mode2(0),
        need_bcast(false), hold_in_lmem(true) {}
  tensor_info_t(slice_info_t slice_info)
      : slice_info(slice_info), mode(TIMESTEP_LDST_UNKNOWN), stage(0),
        use_3ic_opt(0), mode2(0), eu_align(false), need_bcast(false),
        hold_in_lmem(true) {}
  tensor_info_t(Operation *next_op, slice_info_t slice_info)
      : slice_info(slice_info), mode(TIMESTEP_LDST_UNKNOWN), stage(0),
        use_3ic_opt(0), mode2(0), eu_align(false), need_bcast(false),
        hold_in_lmem(true) {
    slice_infos[next_op] = slice_info;
  }
  void add_slice_info(Operation *next_op, slice_info_t slice_info) {
    slice_infos[next_op] = slice_info;
  }

  const std::string mode_str() const {
    switch (mode) {
    case TIMESTEP_LOAD:
      return "TIMESTEP_LOAD";
    case TIMESTEP_STORE:
      return "TIMESTEP_STORE";
    case TIMESTEP_MOVE:
      return "TIMESTEP_MOVE";
    case TIMESTEP_LD_G2L2:
      return "TIMESTEP_LD_G2L2";
    case TIMESTEP_LDST_UNKNOWN:
      return "TIMESTEP_LDST_UNKNOWN";
    }
    return "TIMESTEP_UNKNOWN";
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

  int64_t shape_0;
  int64_t n;
  int64_t c;
  int64_t h;
  int64_t n_slice_num;
  int64_t c_slice_num;
  int64_t h_slice_num;

  shape_secs() {
    nsecs = hsecs = dsecs = wsecs = csecs = shape_0 = 1;
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

  void clear() {
    nsecs = hsecs = dsecs = wsecs = csecs = shape_0 = 1;
    c_slice_num = h_slice_num = n_slice_num = n = c = h = -1;
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
    if (!module::isDebugCmdEnable("detail_info_show")) {
      return;
    }
    llvm::dbgs() << "LgInfo Begin {"
                 << "\n";
    // llvm::dbgs() << "ins"
    //              << "\n";
    // for (auto op : group_ins) {
    //   op.dump();
    // }
    llvm::dbgs() << "ops"
                 << "\n";
    for (auto [index, op] : llvm::enumerate(group_ops)) {
      // op->dump();
      if (op) {
        auto name = module::getName(op).str();
        llvm::dbgs() << llvm::formatv("idx:{0} op: {1}, type: {2}\n", index,
                                      name, op->getName())
                            .str();
      }
    }
    // llvm::dbgs() << "outs"
    //              << "\n";
    // for (auto op : group_outs) {
    //   op.dump();
    // }
    llvm::dbgs() << "} LgInfo End;"
                 << "\n";
  }

  void const dump() const {
    llvm::dbgs() << "============= lg_info =============\n";
    llvm::dbgs() << LOG_KV("base_group_idx", base_group_idx)
                 << LOG_KV("start_idx", start_idx) << LOG_KV("end_idx", end_idx)
                 << LOG_KV("func_start_idx", func_start_idx)
                 << LOG_KV("func_end_idx", func_end_idx)
                 << LOG_KV("cache_key", cache_key) << "\n\n";

    // operations table
    // header
    constexpr int width_0 = 10;
    constexpr int width_1 = 20;
    constexpr int width_2 = 20;
    constexpr int width_3 = 50;
    const char *header_0 = "idx(func)";
    const char *header_1 = "idx(base_group)";
    const char *header_2 = "op_type";
    const char *header_3 = "op_name";
    llvm::dbgs() << llvm::format("%-*s %-*s %-*s %-*s\n", width_0, header_0,
                                 width_1, header_1, width_2, header_2, width_3,
                                 header_3);
    // data table
    for (auto [index, op] : llvm::enumerate(group_ops)) {
      if (op) {
        auto type = op->getName().getStringRef().str();
        auto name = module::getName(op).str();

        llvm::dbgs() << llvm::format(
            "%-*d %-*d %-*s %-*s\n", width_0, func_start_idx + index, width_1,
            start_idx + index, width_2, type.c_str(), width_3, name.c_str());
      }
    }

    llvm::dbgs() << "===================================\n";
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
  int64_t group_id = -1;

  group_valid_type_t is_valid = NOT_CHECK;
  int64_t cache_key = -1;      /**indicate if lg_info is load by cached */
  int64_t base_group_idx = -1; /**cached */
  int64_t start_idx = 0;       /**cached */
  int64_t end_idx = 0;         /**cached */
  int64_t func_start_idx = 0;  /**cached */
  int64_t func_end_idx = 0;    /**cached */
  int64_t group_cost;          /**cached */
  int64_t sort_index =
      0; /**only use for loading and dumping cache, 'index' in json*/

  std::vector<int> free_cores;
  std::map<std::string, std::vector<std::string>> group_banked_tensors;
  int _opt = 2;
};

/* LgCostCache: {key: sub_graph_hash, value:cost }  */
class LgCostCache {
public:
  static LgCostCache &getInstance() {
    static LgCostCache instance;
    return instance;
  }

  // pre encode each local op. Results stored as u64 strings.
  void init(const std::vector<std::vector<Operation *>> &base_groups, bool dynamic_mode) {
    if (dynamic_mode) {
      cache_enabled = false;
      return;
    }
    cache_enabled = true;
    base_group_op_hash.resize(base_groups.size());
    for (size_t idx_group = 0; idx_group < base_groups.size(); ++idx_group) {
      const auto &base_group = base_groups[idx_group];
      base_group_op_hash[idx_group].resize(base_group.size());

      for (size_t idx_op = 0; idx_op < base_group.size(); ++idx_op) {
        base_group_op_hash[idx_group][idx_op] =
            std::to_string(get_op_hash(base_group[idx_op]));
      }
    }
  }

  /// get sub-graph cost from cache
  bool get_cost_from_cache(const uint64_t key, int64_t &cost) {
    if (!cache_enabled) {
      llvm::errs() << "LgCostCache is not enabled.\n";
      return false;
    }
    auto it = cost_cache.find(key);
    if (it != cost_cache.end()) {
      cost = it->second;
      return true;
    }
    return false;
  }

  /// add sub-graph cost to cache
  void add_cache(const uint64_t key, const int64_t cost) {
    cost_cache[key] = cost;
  }

  /// gen hash key for sub-graph
  uint64_t get_graph_hash(const LgInfo &lginfo) {
    /// use relative value-id to serialize sub-graph topo structure.
    llvm::DenseMap<mlir::Value, int> value_id;
    for (auto v : lginfo.group_ins) {
      value_id[v] = value_id.size();
    }
    for (auto v : lginfo.group_outs) {
      value_id[v] = value_id.size();
    }
    for (auto v : lginfo.group_op_outs) {
      if (value_id.find(v) == value_id.end()) {
        value_id[v] = value_id.size();
      }
    }

    const int64_t base_group_idx = lginfo.base_group_idx;
    assert(base_group_idx >= 0); /// -1 for init value.
    const int64_t start_idx = lginfo.start_idx, end_idx = lginfo.end_idx;

    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    os << "LgInfo.type = " << lginfo.type << "\n";
    os << "group_ins:";
    for (auto v : lginfo.group_ins) {
      serialize_value(os, v);
    }
    os << "\ngroup_outs:";
    for (auto v : lginfo.group_outs) {
      serialize_value(os, v);
    }
    os << "\ngroup_ops[" << lginfo.group_ops.size() << "]: ";
    for (int idx = start_idx; idx <= end_idx; ++idx) {
      os << "op: " << base_group_op_hash[base_group_idx][idx]
         << "; operand_relative_ids:";
      // add topo info.
      for (auto v : lginfo.group_ops[idx-start_idx]->getOperands()) {
        if (value_id.find(v) != value_id.end()) {
          os << value_id[v] << " ";
        } else {
          os << "-1 ";
        }
      }
      os << "; ";
    }
    os.flush();
    // std::cout << "\n" << buffer << "\n\n";
    uint64_t key =
        llvm::xxh3_64bits(llvm::StringRef(buffer.data(), buffer.size()));
    return key;
  }

  /// gen hash key for single op
  uint64_t get_op_hash(Operation *op, bool dump = false) {
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    this->serialize_op(os, op);
    if (dump) {
      llvm::outs() << "hash_key: "
                   << llvm::xxh3_64bits(
                          llvm::StringRef(buffer.data(), buffer.size()))
                   << "\n"
                   << "serialized op:" << buffer << "\n\n";
    }
    return llvm::xxh3_64bits(llvm::StringRef(buffer.data(), buffer.size()));
  }

  /// serialize op to string
  void serialize_op(llvm::raw_string_ostream &os, Operation *op) {
    os << op->getName().getStringRef() << ":";
    os << "\tInputs: ";
    for (auto v : op->getOperands()) {
      serialize_value(os, v);
    }
    os << "\tOutputs: ";
    for (auto v : op->getResults()) {
      serialize_value(os, v);
    }
    os << "\tAttrs: ";
    if (auto lg_op = dyn_cast<LocalGenInterface>(op)) {
      lg_op.DumpQuantAgnosticAttrs(os);
    }
    // else {
    //   llvm::errs() << "op is not a LocalGenInterface: \n";
    // }
    os.flush();
  }

  /// serialize value to string
  void serialize_value(llvm::raw_string_ostream &os, Value v) {
    if (v.getType().isa<NoneType>()) {
      os << "None; ";
      return;
    }
    auto shape = module::getShape(v);
    for (auto s : shape) {
      os << s << "x";
    }
    module::getStorageType(v).print(os);
    os << "; ";
  }

public:
  /// a set of u64 strings, for each op in each base_group.
  std::vector<std::vector<std::string>> base_group_op_hash;
  /// group cost cache
  bool cache_enabled = false;
  std::unordered_map<uint64_t, int64_t> cost_cache;

private:
  LgCostCache() = default;
  ~LgCostCache() = default;
  LgCostCache(const LgCostCache &) = delete;
  LgCostCache &operator=(const LgCostCache &) = delete;
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
  LgOptions options_;
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
  bool binary_search_group(bool move_right, const LgOptions &options,
                           std::shared_ptr<dot_graph> dot_graph_log = nullptr);
  std::vector<Operation *> GetParallelNodes(Operation *op);
  void save_result(LgPassIR *pass_ir);
};

struct dag_subnet {
  bool matched = false;
  bool checked = false;
  std::vector<Operation *> ops;
  std::vector<dag_subnet> sub_ops;
};

class speical_layer_group_base {
public:
  speical_layer_group_base() {}
  virtual ~speical_layer_group_base() {}

  virtual bool
  pattern_match_and_parser(Operation *start_op,
                           std::vector<Operation *> &subnet_ops) = 0;
  virtual std::shared_ptr<speical_layer_group_base> clone() = 0;
  virtual std::string name() = 0;
  virtual std::string brief() { return ""; }
  virtual bool convert_to_other_type(
      std::vector<Operation *> &sub_ops,
      std::shared_ptr<speical_layer_group_base> &p_special_grp) = 0;

  bool search_two_mmOp(Operation *start_op, Operation *&next_mmOp,
                       std::vector<Operation *> &subnet_ops);
  void get_batch_size(shape_secs_t &shape_secs);
  bool update_shape_secs_for_ilp_group(shape_secs_t &shape_secs,
                                       const shape_secs_t &max_shape_secs);
  void fill_slice_info(ilp_LgInfo &ilp_lg_info);
  bool inc_slice_num(Operation *op, int &test_slice_n, int &try_c_slice_num,
                     int &try_h_slice_num, int max_n_slice_num,
                     int max_c_slice_num, int max_h_slice_num,
                     int old_target_secs, bool inc_c_slice = true);
  bool inc_n_slice_num(int &n_slice_num, int max_n_slice_num);
  bool is_cut_h(Operation *op);
  bool check_group_valid();
  int get_secs(Operation *op, int slice_n, int c_slice_num, int h_slice_num);
  int get_slice_max_n(int n, int slice_num);
  int get_best_n_slice_num(int n, int slice_num);
  virtual bool CalcMatMulGroupTpNum(ilp_LgInfo &lg_info, Operation *&failed_op,
                                    int64_t core_num);

  Operation *main_mm_op = nullptr;
  std::vector<Operation *> need_del_ops;
  std::vector<Operation *> ops;
  std::vector<Operation *> h_cut_ops;
  std::map<int, int> map_n_slice_num_to_max_n;
  bool col_cut = true;
  bool find_softmax = false;
  bool hdim_is_batch = false;
  std::map<Value, std::vector<int>, value_compare> map_value_to_cut_dims;
};

} // namespace tpu
} // namespace tpu_mlir

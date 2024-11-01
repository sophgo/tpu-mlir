//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupDefs.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

template<typename T1, typename T2>
using Pair = std::pair<T1, T2>;

template<typename T>
using Vector = llvm::SmallVector<T>;

template<typename T>
using Set = std::set<T>;

template<typename T>
using SetVector = llvm::SetVector<T>;

template<typename K, typename V>
using Map = std::map<K, V>;

typedef mlir::detail::ValueImpl* PValue;

// TODO: adapt IndexList to llvm::enumerate
class IndexList {
  llvm::BitVector bitset;
public:
  class iterator {
    int num = 0;
    IndexList* p_index_list;
  public:
    explicit iterator(const IndexList* _p_index_list, int _num) {
      p_index_list = (IndexList*)_p_index_list;
      num = _num;
    }
    iterator& operator++() {
      assert(p_index_list->bitset.test(num));
      if (num < p_index_list->bitset.size())
        num++;
      num = p_index_list->find_balance(num);
      return *this;
    }
    bool operator!=(iterator other) const {
      return num != other.num;
    }
    int operator*() const {
      assert(0 <= num && num < p_index_list->bitset.size());
      return num;
    }
    // iterator traits
    using difference_type = int;
    using value_type = int;
    using pointer = const int*;
    using reference = const int&;
    using iterator_category = std::forward_iterator_tag;
  };

  int size() const {
    return bitset.count();
  }
  void insert(int num) {
    if (num + 1 > bitset.size()) {
      bitset.resize(num + 1);
    }
    bitset.set(num);
  }
  bool contains(int num) const {
    if (num >= size())
      return false;
    return bitset.test(num);
  }
  iterator begin() const {
    const int num = find_balance(0);
    return iterator(this, num);
  }
  iterator end() const {
    return iterator(this, bitset.size());
  }
  void dump() const {
    std::cout << "IndexList with " << size() << " elements = {";
    for (auto i : *this) {
      std::cout << i << ",";
    }
    std::cout << "}" << std::endl;
  }

private:
  // when bitset.test(num), num is under balance
  int find_balance(int num) const {
    for (; num < bitset.size(); num++) {
      if (bitset.test(num)) {
        break;
      }
    }
    return num;
  }
};

class Roster {
public:
  std::set<std::string> expected_input_names;
  std::set<std::string> expected_output_names;
public:
  void clear() {
    expected_input_names.clear();
    expected_output_names.clear();
  }

  bool ins_empty() const {
    return expected_input_names.empty();
  }

  bool outs_empty() const {
    return expected_output_names.empty();
  }

  bool empty() const {
    return ins_empty() && outs_empty();
  }

  void add_input_name(std::string name) {
    expected_input_names.insert(name);
  }

  void add_output_name(std::string name) {
    // if `name` is an input name, ignore it
    if (!is_expected_input(name)) {
      expected_output_names.insert(name);
    }
  }

  void parse_and_add_input_names(std::string inputs) {
    std::stringstream ss;
    std::string name;
    ss.str(inputs);
    while (std::getline(ss, name, ',')) {
      if (!expected_input_names.count(name))
        add_input_name(name);
    }
  }

  void parse_and_add_output_names(std::string outputs) {
    std::stringstream ss;
    std::string name;
    ss.str(outputs);
    while (std::getline(ss, name, ',')) {
      if (!expected_output_names.count(name))
        add_output_name(name);
    }
  }

  bool is_expected_input(std::string name) const {
    return expected_input_names.count(name);
  }

  bool is_expected_output(std::string name) const {
    return expected_output_names.count(name);
  }

  bool is_expected_input(mlir::Value v) const {
    auto name = module::getName(v).str();
    return is_expected_input(name);
  }

  bool is_expected_output(mlir::Value v) const {
    auto name = module::getName(v).str();
    return is_expected_output(name);
  }
};

static ModuleOp g_moduleOp;
static Roster g_roster;

static void copyAddress(Value from, Value to) {
  module::setAddress(to, module::getAddress(from));
}

class MyGmemAllocator {
  int64_t neuron_addr;
  int64_t running_start_addr;
  static std::shared_ptr<MyGmemAllocator> p_gmem_allocator;
public:
  MyGmemAllocator() {
    const int64_t neuron_size = module::getNeuronSize(g_moduleOp);
    neuron_addr = module::getNeuronAddr(g_moduleOp);
    const int64_t coeff_addr = module::getCoeffAddr(g_moduleOp);
    const int64_t coeff_size = module::getCoeffSize(g_moduleOp);
    assert(coeff_addr + coeff_size <= neuron_addr);
    const int64_t io_addr = module::getIOAddr(g_moduleOp);
    const int64_t io_size = module::getIOSize(g_moduleOp);
    assert(io_addr + io_size <= neuron_addr);
    const int64_t start_addr = neuron_addr + neuron_size;
    running_start_addr = start_addr;
  }

  void assign_addr(const Value& v) {
    module::setAddress(v, running_start_addr);
    running_start_addr += module::getBytes(v);
    module::setNeuronSize(g_moduleOp, running_start_addr - neuron_addr);
  }

  static std::shared_ptr<MyGmemAllocator> get() {
    if (!p_gmem_allocator) {
      p_gmem_allocator = std::make_shared<MyGmemAllocator>();
    }
    return p_gmem_allocator;
  }
};

std::shared_ptr<MyGmemAllocator> MyGmemAllocator::p_gmem_allocator = nullptr;

struct upd_info_t {
  IndexList rem_in_idxes;
  IndexList rem_out_idxes;
  Set<PValue> new_ins;
  Set<PValue> new_outs;
};

// return true if op is not func::FuncOp and has at least one block
static bool is_special_blk_op(Operation* op) {
  if (isa<tpu::GroupOp>(op))
    return true;
  return false;
}

static bool is_tmn_op(Operation* op) {
  if (op->hasTrait<OpTrait::IsTerminator>())
    return true;
  return false;
}

// return true when op is an anxilliary operation
static bool is_aux_op(Operation* op) {
  if (isa<top::InputOp>(op))
    return true;
  if (is_tmn_op(op))
    return true;
  if (isa<top::NoneOp, top::WeightOp, tpu::BufferOp>(op))
    return true;
  return false;
}

static Operation* get_terminator(Operation* op) {
  if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
    return funcOp.getFunctionBody().front().getTerminator();
  } else if (auto groupOp = dyn_cast<tpu::GroupOp>(op)) {
    return groupOp.getBody().front().getTerminator();
  } else {
    module::unreachable("unknown blk op");
    return nullptr;
  }
}

static Value get_return_value(Operation* op, int idx) {
  const auto def_op = get_terminator(op);
  return def_op->getOperand(idx);
}

static int get_operand_number(Operation* op, const Value& v) {
  int in_idx = -1;
  for (const auto& [idx, opd] : llvm::enumerate(op->getOperands())) {
    if (opd == v) {
      in_idx = idx;
      break;
    }
  }
  return in_idx;
}

static int get_result_number(Operation* op, const Value& v) {
  int out_idx = -1;
  for (const auto& [idx, res] : llvm::enumerate(op->getResults())) {
    if (res == v) {
      out_idx = idx;
      break;
    }
  }
  return out_idx;
}

static void register_blk_out(Operation* blk_op, upd_info_t& upd_info, const Value& v) {
  auto term_op = get_terminator(blk_op);
  const int out_idx = get_operand_number(term_op, v);
  if (out_idx >= 0) {
    upd_info.rem_out_idxes.insert(out_idx);
  } else {
    upd_info.new_outs.insert(v.getImpl());
  }
}

class GroupDataFlowAnalysis {
public:
  static Operation* analyze_from_tp(Operation* blk_op) {
    auto groupOp = dyn_cast<tpu::GroupOp>(blk_op);
    auto &ops = groupOp.getBody().front().getOperations();
    for (auto& op : ops) {
      if (isa<tpu::YieldOp>(op))
        continue;
      for (auto v : op.getResults()) {
        if (module::isNone(v))
          continue;
        if (g_roster.is_expected_output(v))
          return &op;
      }
    }
    return nullptr;
  }

  static void analyze_from_bt(Operation* blk_op, Operation* end_op_in_blk, upd_info_t& upd_info) {
    SetVector<PValue> live_values;
    for (auto v : end_op_in_blk->getResults()) {
      if (module::isNone(v))
        continue;
      Value u = v;
      for (auto user : v.getUsers()) {
        if (auto storeOp = dyn_cast<tpu::StoreOp>(user)) {
          u = storeOp.getOutput();
          break;
        }
      }
      register_blk_out(blk_op, upd_info, u);
    }
    for (auto v : end_op_in_blk->getOperands()) {
      if (module::isNone(v))
        continue;
      live_values.insert(v.getImpl());
    }
    analyze_from_bt(live_values, upd_info);
  }

  static void analyze_from_bt(Operation* blk_op, int out_idx, upd_info_t& upd_info) {
    upd_info.rem_out_idxes.insert(out_idx);
    auto yield_op = get_terminator(blk_op);
    auto u = yield_op->getOperand(out_idx);
    SetVector<PValue> live_values;
    live_values.insert(u.getImpl());
    analyze_from_bt(live_values, upd_info);
  }

private:
  static void analyze_from_bt(SetVector<PValue>& live_values, upd_info_t& upd_info) {
    for (int i = 0; i < live_values.size(); ++i) {
      auto v = Value(live_values[i]);
      if (module::isNone(v))
        continue;
      auto def_op = v.getDefiningOp();
      if (def_op) {
        if (isa<tpu::LoadOp>(def_op)) {
          auto def_opd = def_op->getOperand(0);
          auto idx = get_operand_number(def_op->getParentOp(), def_opd);
          if (idx >= 0)
            upd_info.rem_in_idxes.insert(idx);
          continue;
        }
        if (g_roster.is_expected_input(v)) {
          upd_info.new_ins.insert(v.getImpl());
          continue;
        }
        for (auto u : def_op->getOperands()) {
          if (module::isNone(u))
            continue;
          live_values.insert(u.getImpl());
        }
      } else {
        llvm_unreachable("unknown error");
      }
    }
  }
};

class BlkDataFlowAnalysis {
public:
  static Operation* analyze_from_tp(Operation* op) {
    if (isa<tpu::GroupOp>(op)) {
      return GroupDataFlowAnalysis::analyze_from_tp(op);
    } else {
      module::unreachable("unknown blk op");
      return nullptr;
    }
  }
  static void analyze_from_bt(Operation* op, Operation* end_op_in_blk, upd_info_t& upd_info) {
    if (isa<tpu::GroupOp>(op)) {
      GroupDataFlowAnalysis::analyze_from_bt(op, end_op_in_blk, upd_info);
    } else {
      module::unreachable("unknown blk op");
    }
  }
  static void analyze_from_bt(Operation* op, int out_idx, upd_info_t& upd_info) {
    if (isa<tpu::GroupOp>(op)) {
      GroupDataFlowAnalysis::analyze_from_bt(op, out_idx, upd_info);
    } else {
      module::unreachable("unknown blk op");
    }
  }
};

class FuncDataFlowAnalysis {
public:
  static Operation* analyze_from_tp(Operation* op) {
    auto funcOp = dyn_cast<func::FuncOp>(op);
    auto &ops = funcOp.getFunctionBody().front().getOperations();
    for (auto& op : ops) {
      if (is_aux_op(&op)) {
        continue;
      }
      if (is_special_blk_op(&op)) {
        Operation* may_be_end_op = BlkDataFlowAnalysis::analyze_from_tp(&op);
        if (may_be_end_op) {
          return may_be_end_op;
        }
      } else {
        for (auto v : op.getResults()) {
          if (module::isNone(v))
            continue;
          if (g_roster.is_expected_output(v)) {
            return &op;
          }
        }
      }
    }
    return nullptr;
  }

  static void analyze_from_bt(SetVector<PValue>& live_values, upd_info_t& sn_upd_info, Map<Operation*, upd_info_t>& blk_upd_infos) {
    for (auto i = 0; i < live_values.size(); ++i) {
      Value v = Value(live_values[i]);
      auto def_op = v.getDefiningOp();
      if (def_op) {
        if (is_aux_op(def_op))
          continue;
        if (g_roster.is_expected_input(v)) {
          sn_upd_info.new_ins.insert(v.getImpl());
          continue;
        }
        if (!is_special_blk_op(def_op)) {
          for (auto def_opd : def_op->getOperands())
            live_values.insert(def_opd.getImpl());
        } else {
          const int out_idx = get_result_number(def_op, v);
          auto& blk_upd_info = blk_upd_infos[def_op];
          BlkDataFlowAnalysis::analyze_from_bt(def_op, out_idx, blk_upd_info);
          for (auto idx : blk_upd_info.rem_in_idxes) {
            auto def_opd = def_op->getOperand(idx);
            live_values.insert(def_opd.getImpl());
          }
        }
      } else if (v.isa<BlockArgument>()) {
        auto idx = v.cast<BlockArgument>().getArgNumber();
        sn_upd_info.rem_in_idxes.insert(idx);
      } else {
        llvm_unreachable("unknown error");
      }
    }
  }
};

static bool need_update(Operation* blk_op, const upd_info_t& blk_upd_info) {
  if (!blk_upd_info.new_ins.empty())
    return true;
  if (!blk_upd_info.new_outs.empty())
    return true;
  if (blk_upd_info.rem_in_idxes.size() < blk_op->getNumOperands())
    return true;
  if (blk_upd_info.rem_out_idxes.size() < blk_op->getNumResults())
    return true;
  return false;
}

static Value create_new_arg(FuncOp& funcOp, Value v, bool b_assign_addr=false) {
  auto& block = funcOp.getFunctionBody().front();
  auto arg = block.addArgument(v.getType(), v.getLoc());
  if (b_assign_addr) {
    MyGmemAllocator::get()->assign_addr(arg);
  }
  funcOp.setType(
    mlir::FunctionType::get(
      funcOp.getContext(),
      block.getArgumentTypes(),
      funcOp.getResultTypes()));
  return arg;
}

static void update_funcOp_ins(func::FuncOp& funcOp) {
  auto& block = funcOp.getFunctionBody().front();
  auto num_args = block.getNumArguments();
  llvm::BitVector idxes_to_del(num_args);
  for (auto i = 0; i < num_args; ++i) {
    if (block.getArgument(i).use_empty()) {
      idxes_to_del.set(i);
    }
  }
  block.eraseArguments(idxes_to_del);
  funcOp.setType(
    mlir::FunctionType::get(
      funcOp.getContext(),
      block.getArgumentTypes(),
      funcOp.getResultTypes()));
}

static void update_funcOp_outs(func::FuncOp& funcOp, const Vector<Value>& outs) {
  auto ret_op = get_terminator(funcOp.getOperation());
  ret_op->setOperands(outs);
  Vector<Type> out_types;
  for (auto pv : outs) {
    out_types.push_back(Value(pv).getType());
  }
  funcOp.setType(
    mlir::FunctionType::get(
      funcOp.getContext(),
      funcOp.getArgumentTypes(),
      out_types));
}

static void update_funcOp_outs(func::FuncOp& funcOp, const IndexList& rem_out_idxes, const Set<PValue>& new_outs) {
  Vector<Value> new_func_outs;
  for (auto idx : rem_out_idxes) {
    auto v = get_return_value(funcOp.getOperation(), idx);
    new_func_outs.push_back(v);
  }
  llvm::append_range(new_func_outs, new_outs);
  update_funcOp_outs(funcOp, new_func_outs);
}

// TODO: needs refactor

template <typename Op>
static void remove_unused_ops(Op& blkOp) {
  auto &ops = blkOp.getBody().front().getOperations();
  Vector<Operation*> all_ops;
  for (auto& op : ops) {
    if (!is_tmn_op(&op))
      all_ops.push_back(&op);
  }
  for (auto iter = all_ops.rbegin(); iter != all_ops.rend(); iter++) {
    if ((*iter)->use_empty()) {
      (*iter)->erase();
    }
  }
}

class GroupIOWorker {
public:
  /**
   * GroupIOWorker could do the following works:
   * 1. create new argument of function directly and link it to LoadOp
   *    * assign address for all of these arguments
   * 2. create new result and link it to StoreOp, which should be passed
   *     to parent function as `new_outs`
   *    * assign address for all of these results
   *    * copy address from these results to outputs of groupOp
   * 3. update links on inputs and outputs of groupOp
   * 4. fix attributes of groupOp, including flows
   */
  static void work(Operation* blk_op, const upd_info_t& blk_upd_info, upd_info_t& sn_upd_info) {
    auto groupOp = dyn_cast<GroupOp>(blk_op);
    const auto live_ops = find_group_live_ops(groupOp, blk_upd_info);
    Vector<Pair<PValue, int64_t>> loads;
    int64_t store_base_id = fix_group_flow(groupOp, blk_upd_info, live_ops, loads);
    Vector<Value> yields;
    auto yield_op = get_terminator(groupOp.getOperation());
    for (auto idx : blk_upd_info.rem_out_idxes) {
      yields.push_back(yield_op->getOperand(idx));
    }
    Vector<Value> new_yields = try_insert_storeOps(blk_upd_info, store_base_id);
    llvm::append_range(yields, new_yields),
    yield_op->setOperands(yields);
    auto new_group_op = reconstruct(groupOp, blk_upd_info, sn_upd_info);
    try_insert_loadOps(new_group_op, blk_upd_info, loads);
    auto num_rem_outs = blk_upd_info.rem_out_idxes.size();
    const auto& new_group_outs = new_group_op->getResults();
    for (auto [idx, v] : llvm::enumerate(new_group_outs)) {
      if (idx < num_rem_outs) {
        auto term_op = get_terminator(new_group_op->getParentOp());
        const int out_idx = get_operand_number(term_op, v);
        if (out_idx >= 0) {
          sn_upd_info.rem_out_idxes.insert(out_idx);
          continue;
        }
      }
      sn_upd_info.new_outs.insert(v.Value::getImpl());
    }
    auto new_groupOp = dyn_cast<tpu::GroupOp>(new_group_op);
    remove_unused_ops(new_groupOp);
  }

private:
  static LayerGroupAttr make_lg_param(const LayerGroupAttr& ginfo, int64_t stage, int64_t id) {
    auto ctx = g_moduleOp.getContext();
    return LayerGroupAttr::get(ctx,
                               ginfo.getOutAddr(),
                               ginfo.getOutSize(),
                               0, 0,
                               ginfo.getEuAlign(),
                               false,
                               ginfo.getInHsliceOffset(),
                               ginfo.getNIdx(),
                               ginfo.getNSlice(),
                               ginfo.getCIdx(),
                               ginfo.getCSlice(),
                               ginfo.getDIdx(),
                               ginfo.getDSlice(),
                               ginfo.getHIdx(),
                               ginfo.getHSlice(),
                               ginfo.getWIdx(),
                               ginfo.getWSlice(),
                               id,
                               stage,
                               0,
                               ginfo.getGroupType());
  }

  static LayerGroupAttr get_ginfo(Operation* op) {
    return op->getAttr("ginfo").cast<LayerGroupAttr>();
  }

  static void set_LgOp_id(Operation* op, int64_t id) {
    const LayerGroupAttr ginfo = get_ginfo(op);
    auto ctx = g_moduleOp.getContext();
    auto param = LayerGroupAttr::get(ctx,
                                    ginfo.getOutAddr(),
                                    ginfo.getOutSize(),
                                    ginfo.getBufferAddr(),
                                    ginfo.getBufferSize(),
                                    ginfo.getEuAlign(),
                                    ginfo.getCanMerge(),
                                    ginfo.getInHsliceOffset(),
                                    ginfo.getNIdx(),
                                    ginfo.getNSlice(),
                                    ginfo.getCIdx(),
                                    ginfo.getCSlice(),
                                    ginfo.getDIdx(),
                                    ginfo.getDSlice(),
                                    ginfo.getHIdx(),
                                    ginfo.getHSlice(),
                                    ginfo.getWIdx(),
                                    ginfo.getWSlice(),
                                    id,
                                    ginfo.getStage(),
                                    ginfo.getSliceIdx(),
                                    ginfo.getGroupType());
    op->setAttr(LocalGenInterface::kLayerGroupAttrName,
                param);
  }

  static int get_LgOp_id(Operation* op) {
    const LayerGroupAttr ginfo = get_ginfo(op);
    return ginfo.getId();
  }

  static void try_insert_loadOps(Operation* group_op, const upd_info_t& blk_upd_info, const Vector<Pair<PValue, int64_t>>& loads) {
    const auto base_idx = blk_upd_info.rem_in_idxes.size();
    const auto ctx = g_moduleOp.getContext();
    OpBuilder builder(ctx);
    for (auto [idx, pair] : llvm::enumerate(loads)) {
      auto [pv, id] = pair;
      auto v = Value(pv);
      builder.setInsertionPointAfterValue(v);
      auto opd = group_op->getOperand(base_idx + idx);
      Vector<Value> operands({opd});
      std::string name = module::getName(v).str();
      Vector<NamedAttribute> attrs;
      LayerGroupAttr ginfo = get_ginfo(v.getDefiningOp());
      attrs.push_back(builder.getNamedAttr(
        "lmem_type", builder.getI64IntegerAttr(LMEM_ACTIVATION)));
      attrs.push_back(
        builder.getNamedAttr(
          LocalGenInterface::kLayerGroupAttrName,
          make_lg_param(ginfo, 0 /*STAGE_LOAD*/, id)
        )
      );
      auto loadOp = builder.create<tpu::LoadOp>(
        NameLoc::get(builder.getStringAttr("load_" + name)),
        v.getType(), operands, attrs);
      v.replaceAllUsesWith(loadOp.getOutput());
    }
  }

  static Vector<Value> try_insert_storeOps(const upd_info_t& blk_upd_info, int64_t base_id) {
    const auto ctx = g_moduleOp.getContext();
    OpBuilder builder(ctx);
    Vector<Value> new_yields;
    int64_t id = base_id;
    for (auto& pv : blk_upd_info.new_outs) {
      auto v = Value(pv);
      auto op = v.getDefiningOp();
      builder.setInsertionPointAfter(op);
      Vector<Value> operands;
      operands.push_back(v);
      operands.push_back(module::getNoneOp(op));
      std::string name = module::getName(v).str();
      Vector<NamedAttribute> attrs;
      LayerGroupAttr ginfo = get_ginfo(op);
      attrs.push_back(
        builder.getNamedAttr(
          LocalGenInterface::kLayerGroupAttrName,
          make_lg_param(ginfo, 2 /*STAGE_STORE*/, id)
        )
      );
      auto storeOp = builder.create<tpu::StoreOp>(
        NameLoc::get(builder.getStringAttr(name)),
                     v.getType(), operands, attrs);
      auto store_out = storeOp.getOutput();
      MyGmemAllocator::get()->assign_addr(store_out);
      new_yields.push_back(store_out);
      id++;
    }
    return new_yields;
  }

  static int64_t get_group_ts_rows(tpu::GroupOp groupOp, Vector<Vector<int64_t>>& ts_rows) {
    auto flow = module::getI64Array(groupOp.getFlow());
    int64_t max_id = 0;
    Vector<int64_t> ts_row;
    for (auto i = 1; i < flow->size(); ++i) {
      if (flow->at(i) < 0) {
        ts_rows.push_back(ts_row);
        ts_row.clear();
        continue;
      }
      ts_row.push_back(flow->at(i));
      max_id = std::max(max_id, flow->at(i));
    }
    ts_rows.push_back(ts_row);
    return max_id;
  }

  static void set_group_ts_rows(tpu::GroupOp groupOp, const Vector<Vector<int64_t>>& ts_rows) {
    const auto ctx = g_moduleOp.getContext();
    OpBuilder builder(ctx);
    Vector<int64_t> flow;
    for (auto i = 0; i < ts_rows.size(); ++i) {
      flow.push_back(-i);
      flow.insert(flow.end(), ts_rows[i].begin(), ts_rows[i].end());
    }
    groupOp->setAttr("flow", builder.getI64ArrayAttr(flow));
  }

  static Vector<Operation*> find_group_live_ops(tpu::GroupOp groupOp, const upd_info_t& blk_upd_info) {
    SetVector<Operation*> _live_ops;
    auto yield_op = get_terminator(groupOp.getOperation());
    for (auto idx : blk_upd_info.rem_out_idxes) {
      _live_ops.insert((yield_op->getOperand(idx)).getDefiningOp());
    }
    for (auto pv : blk_upd_info.new_outs) {
      _live_ops.insert(Value(pv).getDefiningOp());
    }
    for (int i = 0; i < _live_ops.size(); ++i) {
      auto op = _live_ops[i];
      for (auto v : op->getOperands()) {
        if (blk_upd_info.new_ins.count(v.getImpl()))
          continue;
        auto def_op = v.getDefiningOp();
        if (def_op) {
          _live_ops.insert(def_op);
        }
      }
    }
    Vector<Operation*> live_ops;
    auto &ops = groupOp.getBody().front().getOperations();
    for (auto& op : ops) {
      if (_live_ops.count(&op))
        live_ops.push_back(&op);
    }
    return live_ops;
  }

  static int64_t fix_group_flow(tpu::GroupOp groupOp, const upd_info_t& blk_upd_info,
                                const Vector<Operation*>& live_ops,
                                Vector<Pair<PValue, int64_t>>& loads) {
    Vector<Vector<int64_t>> ts_rows;
    get_group_ts_rows(groupOp, ts_rows);
    // [old_id : new_id]
    Map<int64_t, int64_t> id_map;
    // [old_id : inc]
    Map<int64_t, int64_t> id_inc;
    Set<PValue> vis_vals;
    loads.clear();
    // step1 : fix id of live ops - assign new id by topo order
    int64_t new_id = 0;
    for (auto op : live_ops) {
      int64_t inc = 0;
      for (Value v : op->getOperands()) {
        auto pv = v.getImpl();
        if (blk_upd_info.new_ins.count(pv) && !vis_vals.count(pv)) {
          vis_vals.insert(pv);
          loads.emplace_back(pv, new_id);
          inc++;
          new_id++;
        }
      }
      const auto old_id = get_LgOp_id(op);
      if (inc) {
        id_inc[old_id] = inc;
      }
      set_LgOp_id(op, new_id);
      id_map[old_id] = new_id;
      new_id++;
    }
    // step2 : fix flow
    MapVector<int64_t, Vector<int64_t>> inserts;
    for (auto i = 0; i < ts_rows.size(); ++i) {
      for (auto j = 0; j < ts_rows[i].size(); ++j) {
        auto id = ts_rows[i][j];
        if (!id_map.count(id)) {
          ts_rows[i].erase(ts_rows[i].begin() + j);
          j--;
        } else {
          auto new_id = id_map[id];
          if (id_inc.count(id)) {
            auto inc = id_inc[id];
            Vector<int64_t> ld_ts_row;
            for (auto i = new_id - inc; i < new_id; ++i) {
              ld_ts_row.push_back(i);
            }
            inserts[i + 1] = ld_ts_row;
          }
          ts_rows[i][j] = new_id;
        }
      }
      if (ts_rows[i].empty()) {
        ts_rows.erase(ts_rows.begin() + i);
        i--;
      }
    }
    for (auto it = inserts.rbegin(); it != inserts.rend(); ++it) {
      ts_rows.insert(ts_rows.begin() + it->first, it->second);
    }
    const int num_new_st = blk_upd_info.new_outs.size();
    if (num_new_st) {
      Vector<int64_t> st_ts_row;
      for (auto i = new_id; i < new_id + num_new_st; ++i) {
        st_ts_row.push_back(i);
      }
      ts_rows.insert(ts_rows.begin(), st_ts_row);
    }
    set_group_ts_rows(groupOp, ts_rows);
    return new_id;
  }

  static Operation* reconstruct(tpu::GroupOp groupOp, const upd_info_t& blk_upd_info, upd_info_t& sn_upd_info) {
    const auto ctx = g_moduleOp.getContext();
    OpBuilder builder(ctx);
    Vector<Value> group_new_ins;
    for (auto idx : blk_upd_info.rem_in_idxes) {
      group_new_ins.push_back(groupOp->getOperand(idx));
    }
    auto parent_op = groupOp->getParentOp();
    assert(isa<func::FuncOp>(parent_op));
    auto funcOp = dyn_cast<func::FuncOp>(parent_op);
    for (auto& v : blk_upd_info.new_ins) {
      auto arg = create_new_arg(funcOp, v, true);
      group_new_ins.push_back(arg);
      sn_upd_info.new_ins.insert(arg.getImpl());
    }
    Vector<Type> new_out_types;
    Vector<Location> locs;
    for (auto idx : blk_upd_info.rem_out_idxes) {
      const Value out = groupOp->getResult(idx);
      new_out_types.push_back(out.getType());
      locs.push_back(module::getLoc(out));
    }
    for (auto pout : blk_upd_info.new_outs) {
      auto out = Value(pout);
      new_out_types.push_back(out.getType());
      locs.push_back(module::getLoc(out));
    }
    const auto group_loc = builder.getFusedLoc(locs);
    builder.setInsertionPoint(groupOp);
    auto new_groupOp =
      builder.create<tpu::GroupOp>(
        group_loc, new_out_types, group_new_ins, groupOp->getAttrs());
    new_groupOp.getBody().takeBody(groupOp.getBody());
    // replace all uses of from operand of groupOp to that of new_groupOp
    int new_idx = 0;
    for (auto idx : blk_upd_info.rem_out_idxes) {
      groupOp->getResult(idx).replaceAllUsesWith(new_groupOp->getResult(new_idx));
      new_idx++;
    }
    // inhert addr from yield
    auto yield_op = get_terminator(new_groupOp.getOperation());
    for (auto idx = new_idx; idx < yield_op->getNumOperands(); ++idx) {
      copyAddress(yield_op->getOperand(idx), new_groupOp->getResult(idx));
    }
    return new_groupOp.getOperation();
  }
};

class BlkIOWorker {
public:
  /**
   * BlkIOWorker could do the following works:
   * 1. create new argument of function directly and link it to load related op
   * 2. create new result and link it to store related op, which should be passed
   *     to parent function as `new_outs`
   * 3. update links on inputs and outputs of blk Op
   */
  static void work(Operation* blk_op, const upd_info_t& blk_upd_info, upd_info_t& sn_upd_info) {
    if (!need_update(blk_op, blk_upd_info))
      return;
    if (isa<GroupOp>(blk_op)) {
      GroupIOWorker::work(blk_op, blk_upd_info, sn_upd_info);
    } else {
      module::unreachable("unknown blk op");
    }
  }
};

class MultiSubnetTruncIOWorker {
  // [subnet_id, io_id]
  typedef Pair<int, int> id_pair_t;
  Operation* end_op_in_blk = nullptr;
  // callOps that meets
  Vector<Operation*> calls;
  // subnet funcOps that meets
  Vector<Operation*> subnets;
  // [subnet_id : io_upd_info]
  Map<int, upd_info_t> sn_upd_infos;
  // [subnet_id : value]
  Map<int, SetVector<PValue>> live_values;
  // [use_subnet_id : [out_idx : (def_subnet_id, in_idx)]]
  Map<int, Map<int, id_pair_t>> subnet_io_maps;

public:
  void invoke() {
    auto mainFuncOp = module::getMainFuncOp(g_moduleOp);
    analyze_call_data_flow(mainFuncOp);
    bool found = false;
    auto max_subnet_id = 0;
    for (; max_subnet_id < subnets.size(); ++max_subnet_id) {
      found = analyze_subnet_data_flow_from_tp(max_subnet_id);
      if (found)
        break;
    }
    if (!found)
      return;
    work_for_subnet(max_subnet_id);
    for (auto subnet_id = max_subnet_id - 1; subnet_id >= 0; --subnet_id) {
      if (live_values[subnet_id].empty())
        continue;
      work_for_subnet(subnet_id);
    }
    work_for_main();
  }

private:
  void analyze_call_data_flow(func::FuncOp mainOp) {
    calls.clear();
    subnets.clear();
    Map<PValue, id_pair_t> id_pairs;
    mainOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        auto cid = calls.size();
        for (auto [iid, v] : llvm::enumerate(op->getOperands())) {
          auto pv = v.getImpl();
          if (id_pairs.count(pv)) {
            subnet_io_maps[cid][iid] = id_pairs[pv];
          }
        }
        for (auto [oid, v] : llvm::enumerate(op->getResults())) {
          auto pv = v.Value::getImpl();
          id_pairs[pv] = id_pair_t{(int)cid, (int)oid};
        }
        calls.push_back(callOp.getOperation());
        auto funcOp = module::getFuncOp(g_moduleOp, callOp.getCallee());
        subnets.push_back(funcOp.getOperation());
      }
      return WalkResult::advance();
    });
  }

  bool analyze_subnet_data_flow_from_tp(int subnet_id) {
    auto func_op = subnets[subnet_id];
    Operation* may_be_end_op = FuncDataFlowAnalysis::analyze_from_tp(func_op);
    if (may_be_end_op == nullptr)
      return false;
    if (is_special_blk_op(may_be_end_op->getParentOp())) {
      end_op_in_blk = may_be_end_op;
    } else {
      for (auto v : may_be_end_op->getResults()) {
        if (module::isNone(v))
          continue;
        live_values[subnet_id].insert(v.Value::getImpl());
        register_blk_out(func_op, sn_upd_infos[subnet_id], v);
      }
    }
    return true;
  }

  void analyze_subnet_data_flow_from_bt(int subnet_id, Map<Operation*, upd_info_t>& blk_upd_infos) {
    auto& sn_upd_info = sn_upd_infos[subnet_id];
    FuncDataFlowAnalysis::analyze_from_bt(live_values[subnet_id], sn_upd_info, blk_upd_infos);
    auto& subnet_io_map = subnet_io_maps[subnet_id];
    for (auto iid : sn_upd_info.rem_in_idxes) {
      if (!subnet_io_map.count(iid))
        continue;
      auto [def_subnet_id, io_id] = subnet_io_map[iid];
      auto def_subnet = subnets[def_subnet_id];
      auto ret_value = get_return_value(def_subnet, io_id);
      live_values[def_subnet_id].insert(ret_value.getImpl());
      sn_upd_infos[def_subnet_id].rem_out_idxes.insert(io_id);
    }
  }

  void work_for_subnet(int subnet_id) {
    Map<Operation*, upd_info_t> blk_upd_infos;
    if (end_op_in_blk) {
      auto def_op = end_op_in_blk->getParentOp();
      auto& blk_upd_info = blk_upd_infos[def_op];
      BlkDataFlowAnalysis::analyze_from_bt(def_op, end_op_in_blk, blk_upd_info);
      for (auto idx : blk_upd_info.rem_in_idxes) {
        auto opd = def_op->getOperand(idx);
        live_values[subnet_id].insert(opd.getImpl());
      }
      end_op_in_blk = nullptr;
    }
    analyze_subnet_data_flow_from_bt(subnet_id, blk_upd_infos);
    auto subnet = subnets[subnet_id];
    auto funcOp = dyn_cast<func::FuncOp>(subnet);
    upd_info_t& sn_upd_info = sn_upd_infos[subnet_id];
    auto &ops = funcOp.getBody().front().getOperations();
    Set<PValue> new_ins;
    for (auto pv : sn_upd_info.new_ins) {
      auto v = Value(pv);
      auto arg = create_new_arg(funcOp, v);
      v.replaceAllUsesWith(arg);
      new_ins.insert(arg.getImpl());
    }
    sn_upd_info.new_ins = new_ins;
    for (auto& op : ops) {
      if (is_special_blk_op(&op) && blk_upd_infos.count(&op))
        BlkIOWorker::work(&op, blk_upd_infos[&op], sn_upd_info);
    }
    update_funcOp_outs(funcOp, sn_upd_info.rem_out_idxes, sn_upd_info.new_outs);
    remove_unused_ops(funcOp);
    update_funcOp_ins(funcOp);
  }

  // find subnet_io_maps for new groups of callOps
  void update_subnet_io_maps_for_main(const Vector<int>& live_subnet_ids) {
    // [subnet_id : [old_idx : new_idx]]
    Map<int, Map<int, int>> in_idx_upd_maps, out_idx_upd_maps;
    auto find_idx_upd_map = [](const IndexList& idxes, Map<int, int>& idx_upd_map) {
      int new_idx = 0;
      for (auto old_idx : idxes) {
        idx_upd_map[old_idx] = new_idx;
        new_idx++;
      }
    };
    for (auto subnet_id : live_subnet_ids) {
      find_idx_upd_map(sn_upd_infos[subnet_id].rem_in_idxes, in_idx_upd_maps[subnet_id]);
      find_idx_upd_map(sn_upd_infos[subnet_id].rem_out_idxes, out_idx_upd_maps[subnet_id]);
    }
    for (auto& [use_subnet_id, old_maps] : subnet_io_maps) {
      auto& out_idx_upd_map = out_idx_upd_maps[use_subnet_id];
      Map<int, id_pair_t> new_maps;
      for (auto [out_idx, id_pair] : old_maps) {
        auto new_out_idx = out_idx_upd_map[out_idx];
        auto new_in_idx = in_idx_upd_maps[id_pair.first][id_pair.second];
        new_maps[new_out_idx] = std::make_pair(id_pair.first, new_in_idx);
      }
      subnet_io_maps[use_subnet_id] = new_maps;
    }
  }

  void work_for_main() {
    // get live subnet ids by ascending order
    Vector<int> live_subnet_ids;
    const int max_subnet_id = subnets.size();
    for (auto id = 0; id < max_subnet_id; ++id) {
      if (sn_upd_infos.count(id))
        live_subnet_ids.push_back(id);
    }
    llvm::sort(live_subnet_ids);
    const auto ctx = g_moduleOp.getContext();
    OpBuilder builder(ctx);
    auto mainFuncOp = module::getMainFuncOp(g_moduleOp);
    auto ret_op = get_terminator(mainFuncOp.getOperation());
    builder.setInsertionPoint(ret_op);
    // update_subnet_io_maps_for_main(live_subnet_ids);
    for (auto subnet_id : live_subnet_ids) {
      auto call_op = calls[subnet_id];
      auto old_subnet_ins = call_op->getOperands();
      auto& sn_upd_info = sn_upd_infos[subnet_id];
      Vector<Value> new_call_ins;
      for (auto idx : sn_upd_info.rem_in_idxes) {
        new_call_ins.push_back(old_subnet_ins[idx]);
      }
      for (auto pv : sn_upd_info.new_ins) {
        auto v = Value(pv);
        const auto& arg = create_new_arg(mainFuncOp, v);
        auto inputOp = builder.create<top::InputOp>(
          v.getLoc(), v.getType(), ValueRange{arg});
        new_call_ins.push_back(inputOp.getOutput());
      }
      auto new_callOp = builder.create<func::CallOp>(
        builder.getUnknownLoc(),
        dyn_cast<func::FuncOp>(subnets[subnet_id]),
        new_call_ins);
      // replace all uses of from operand of groupOp to that of new_groupOp
      int new_idx = 0;
      for (auto idx : sn_upd_info.rem_out_idxes) {
        call_op->getResult(idx).replaceAllUsesWith(new_callOp->getResult(new_idx));
        new_idx++;
      }
      calls[subnet_id] = new_callOp.getOperation();
    }
    update_funcOp_outs(mainFuncOp, calls[max_subnet_id - 1]->getResults());
    remove_unused_ops(mainFuncOp);
    update_funcOp_ins(mainFuncOp);
  }
};

class TruncIOWorker {
public:
  void invoke() {
    upd_info_t sn_upd_info;
    SetVector<PValue> live_values;
    Map<Operation*, upd_info_t> blk_upd_infos;
    auto mainFuncOp = module::getMainFuncOp(g_moduleOp);
    auto main_func_op = mainFuncOp.getOperation();
    Operation* may_be_end_op = FuncDataFlowAnalysis::analyze_from_tp(main_func_op);
    if (may_be_end_op == nullptr)
      return;
    for (auto v : may_be_end_op->getResults()) {
      if (module::isNone(v))
        continue;
      live_values.insert(v.Value::getImpl());
      register_blk_out(main_func_op, sn_upd_info, v);
    }
    FuncDataFlowAnalysis::analyze_from_bt(live_values, sn_upd_info, blk_upd_infos);
    Set<PValue> new_ins;
    const auto ctx = g_moduleOp.getContext();
    OpBuilder builder(ctx);
    for (auto pv : sn_upd_info.new_ins) {
      auto v = Value(pv);
      auto arg = create_new_arg(mainFuncOp, v);
      builder.setInsertionPointAfterValue(arg);
      auto inputOp = builder.create<top::InputOp>(
        v.getLoc(), v.getType(), ValueRange{arg});
      v.replaceAllUsesWith(inputOp.getOutput());
      new_ins.insert(arg.getImpl());
    }
    sn_upd_info.new_ins = new_ins;
    update_funcOp_outs(mainFuncOp, sn_upd_info.rem_out_idxes, sn_upd_info.new_outs);
    remove_unused_ops(mainFuncOp);
    update_funcOp_ins(mainFuncOp);
  }
};

/**
 * From top to bottom, find first op END_OP whose one output matches the expected output.
 *  Set the network outputs as the outputs of END_OP which matches.
 * From bottom to top, find last op START_OP whose one input matches the expected input.
 *  Set the network inputs as the outputs of START_OP which matches.
 *  Note that the number of START_OP is usually more than one.
 */
class TruncIOPass : public TruncIOBase<TruncIOPass> {
public:
  TruncIOPass() {}

  void update_module_weights(ModuleOp& mOp) {
    std::string old_wfile_name = module::getWeightFileAttr().str();
    std::string postfix = "weight.npz";
    std::size_t pos = old_wfile_name.find(postfix);
    if (pos == -1)
      module::unreachable("weight file name error");
    std::string new_wfile_name = old_wfile_name.substr(0, pos) + "_trunc_" + postfix;
    auto wFile = std::make_unique<mlir::TensorFile>(old_wfile_name, false);
    std::set<StringRef> npz_names;
    wFile->getAllNames(npz_names);
    std::set<StringRef> weight_names;
    for (auto func : mOp.getOps<FuncOp>()) {
      func.walk([&](top::WeightOp op) {
        weight_names.insert(module::getName(op.getOperation()));
      });
    }
    std::set<StringRef> del_names;
    for (auto name : npz_names) {
      if (!weight_names.count(name)) {
        del_names.insert(name);
      }
    }
    for (auto &name : del_names) {
      wFile->deleteTensor(name);
    }
    wFile->save(new_wfile_name);
    module::setWeightFileAttr(new_wfile_name);
  }

  void update_module_io_names(ModuleOp& mOp) {
    std::vector<Value> inputs, outputs;
    module::getInputsOutputs(mOp, inputs, outputs);
    llvm::SmallVector<StringRef> input_names, output_names;
    for (auto in : inputs)
      input_names.push_back(module::getName(in));
    for (auto out : outputs)
      output_names.push_back(module::getName(out));
    module::setInputs(input_names);
    module::setOutputs(output_names);
  }

  std::vector<StringRef> getNetInputs(ModuleOp& mOp) {
    if (!module::isSubnetDividedState()) {
      auto mainFuncOp = module::getMainFuncOp(mOp);
      auto& ops = mainFuncOp.getBody().front().getOperations();
      std::vector<StringRef> inputs;
      for (auto& op : ops) {
        if (auto inputOp = dyn_cast<top::InputOp>(&op)) {
          inputs.push_back(module::getName(inputOp.getOutput()));
        }
      }
      return inputs;
    } else {
      return *module::getInputs();
    }
  }

  void runOnOperation() override {
    auto mOp = getOperation();
    module::init(mOp);
    g_roster.clear();
    Set<std::string> expected_input_names;
    auto _input_names = getNetInputs(mOp);
    for (auto name: _input_names) {
      g_roster.add_input_name(name.str());
    }
    g_roster.parse_and_add_input_names(this->inputs);
    g_roster.parse_and_add_output_names(this->outputs);
    if (g_roster.outs_empty())
      return;
    if (!module::isSubnetDividedState()) {
      g_moduleOp = mOp;
      TruncIOWorker().invoke();
    } else {
      g_moduleOp = module::getAllModules()->at(0);
      MultiSubnetTruncIOWorker().invoke();
      update_module_io_names(g_moduleOp);
    }
    update_module_weights(g_moduleOp);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createTruncIOPass() {
  return std::make_unique<TruncIOPass>();
}
} // namespace tpu
} // namespace tpu_mlir

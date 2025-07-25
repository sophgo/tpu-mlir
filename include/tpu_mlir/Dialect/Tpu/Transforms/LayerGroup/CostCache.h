//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupDefs.h"
#include "tpu_mlir/Interfaces/LocalGenInterface.h"

namespace tpu_mlir {
namespace tpu {

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

} // namespace tpu
} // namespace tpu_mlir

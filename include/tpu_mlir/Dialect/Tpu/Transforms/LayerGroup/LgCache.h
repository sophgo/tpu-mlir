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

struct LgCacheKey {
  std::string structure_hash_str;
  std::string cost_hash_str;

  bool operator==(const LgCacheKey &other) const {
    return structure_hash_str == other.structure_hash_str &&
           cost_hash_str == other.cost_hash_str;
  }
};

/* LgCache: {key: sub_graph_hash, value:cost }  */
class LgCache {
public:
  static LgCache &getInstance() {
    static LgCache instance;
    return instance;
  }

  void init(const llvm::SetVector<Operation *> &subnet_ops, bool dynamic_mode) {
    hash_counts.clear();
    hash_counts_map.clear();
    hash_op_map.clear();
    if (dynamic_mode) {
      cache_enabled = false;
      return;
    }
    cache_enabled = true;
    subnet_op_hash.resize(subnet_ops.size());
    for (int64_t idx_op = 0; idx_op < subnet_ops.size(); ++idx_op) {
      LgCacheKey op_hash;
      auto op_structure_hash = get_op_hash(subnet_ops[idx_op], true);
      auto op_cost_hash = get_op_hash(subnet_ops[idx_op]);
      op_hash.structure_hash_str = std::to_string(op_structure_hash);
      op_hash.cost_hash_str = std::to_string(op_cost_hash);
      subnet_op_hash[idx_op] = op_hash;

      // structure hash counting
      hash_op_map[op_structure_hash].push_back(idx_op);
      if (hash_counts.find(op_structure_hash) == hash_counts.end()) {
        hash_counts[op_structure_hash] = 1;
      } else {
        hash_counts[op_structure_hash] += 1;
      }
    }
    for (int64_t idx_op = 0; idx_op < subnet_ops.size(); ++idx_op) {
      auto op_structure_hash = get_op_hash(subnet_ops[idx_op], true);
      auto cnt = hash_counts[op_structure_hash];
      hash_counts_map[cnt].insert(op_structure_hash);
    }
    DEBUG_WITH_TYPE("layer_group_cache", {
      llvm::dbgs() << "LgCache initialized. op hash counts:\n";
      // print op hash counts according to base_groups
      for (size_t idx_op = 0; idx_op < subnet_ops.size(); ++idx_op) {
        auto op_structure_hash = get_op_hash(subnet_ops[idx_op], true);
        auto op_cost_hash = get_op_hash(subnet_ops[idx_op]);
        llvm::dbgs() << "  Op " << idx_op << ": "
                     << "cost_hash = " << op_cost_hash
                     << ", structure_hash = " << op_structure_hash
                     << ", count = " << hash_counts[op_structure_hash] << "\n";
        subnet_ops[idx_op]->dump();
        llvm::dbgs() << "\n";
      }
    });
  }

  std::unordered_map<uint64_t, int64_t> get_hash_counts() {
    return hash_counts;
  }

  std::map<int64_t, std::set<uint64_t>, std::less<int64_t>>
  get_hash_counts_map() {
    return hash_counts_map;
  }

  std::unordered_map<uint64_t, std::vector<int64_t>> get_hash_op_map() {
    return hash_op_map;
  }

  /// get sub-graph cost from cache
  bool get_info_from_cost_cache(const uint64_t key, int64_t &cost,
                                shape_secs_t &shape_secs) {
    if (!cache_enabled) {
      llvm::errs() << "LgCache is not enabled.\n";
      return false;
    }
    auto cost_it = cost_cache.find(key);
    if (cost_it != cost_cache.end()) {
      cost = cost_it->second;
      auto shape_secs_it = shape_secs_cache.find(key);
      if (shape_secs_it != shape_secs_cache.end()) {
        shape_secs = shape_secs_it->second;
      } else {
        shape_secs.clear();
      }
      return true;
    }
    return false;
  }

  bool get_cost_from_cost_cache(const uint64_t key, int64_t &cost,
                                int shape_secs_search_level) {
    if (!cache_enabled) {
      llvm::errs() << "LgCache is not enabled.\n";
      return false;
    }
    auto search_level_it = search_level_cache.find(key);
    if (search_level_it == search_level_cache.end() ||
        search_level_it->second < shape_secs_search_level) {
      // llvm::dbgs() << "shape_secs_search_level: " << shape_secs_search_level
      // << "\n"; llvm::dbgs() << "cache miss\n";
      return false;
    }
    auto cost_it = cost_cache.find(key);
    if (cost_it != cost_cache.end()) {
      cost = cost_it->second;
      // llvm::dbgs() << "cache hit\n";
      return true;
    }
    // llvm::dbgs() << "cache miss\n";
    return false;
  }

  bool get_shape_secs_from_cost_cache(const uint64_t key,
                                      shape_secs_t &shape_secs) {
    if (!cache_enabled) {
      llvm::errs() << "LgCache is not enabled.\n";
      return false;
    }
    auto shape_secs_it = shape_secs_cache.find(key);
    if (shape_secs_it != shape_secs_cache.end()) {
      shape_secs = shape_secs_it->second;
      return true;
    }
    return false;
  }

  /// add sub-graph cost to cache
  void add_cost_cache(const uint64_t key, const LgInfo &lg_info) {
    cost_cache[key] = lg_info.group_cost;
    shape_secs_cache[key] = lg_info.shape_secs;
    search_level_cache[key] = lg_info.shape_secs_search_level;
    // llvm::dbgs() << "add_cost_cache: key = " << key
    //              << "; cost = " << lg_info.group_cost
    //              << "; shape_secs_search_level = " <<
    //              lg_info.shape_secs_search_level
    //              << "\n";
  }

  /// gen hash key for sub-graph
  bool get_graph_hash(LgInfo &lginfo, uint64_t &hash_key,
                      bool quant_agnostic = false) {
    if (!cache_enabled) {
      return false;
    }
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

    const int64_t func_start_idx = lginfo.func_start_idx,
                  func_end_idx = lginfo.func_end_idx;
    if (func_start_idx < 0 || func_end_idx < 0) {
      return false;
    }
    if (subnet_op_hash.size() <= func_end_idx ||
        subnet_op_hash.size() <= func_start_idx) {
      return false;
    }

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
    for (int idx = func_start_idx; idx <= func_end_idx; ++idx) {
      std::string current_op_hash;
      if (quant_agnostic) {
        current_op_hash = subnet_op_hash[idx].structure_hash_str;
      } else {
        current_op_hash = subnet_op_hash[idx].cost_hash_str;
      }
      os << "op: " << current_op_hash << "; operand_relative_ids:";
      // add topo info.
      for (auto v : lginfo.group_ops[idx - func_start_idx]->getOperands()) {
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
    hash_key = llvm::xxh3_64bits(llvm::StringRef(buffer.data(), buffer.size()));
    return true;
  }

  /// gen hash key for single op
  /// if quant_agnostic is true, ignore quantization attributes
  uint64_t get_op_hash(Operation *op, bool quant_agnostic = false,
                       bool dump = false) {
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
  void serialize_op(llvm::raw_string_ostream &os, Operation *op,
                    bool quant_agnostic = false) {
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
      lg_op.DumpAttrs(os, quant_agnostic);
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
  bool cache_enabled = true;
  std::vector<LgCacheKey> subnet_op_hash;
  /// group cost cache
  std::unordered_map<uint64_t, int64_t> cost_cache;
  std::unordered_map<uint64_t, int> search_level_cache;
  std::unordered_map<uint64_t, shape_secs_t> shape_secs_cache;
  // structure cache
  std::unordered_map<uint64_t, int64_t> structure_cache;
  std::unordered_map<uint64_t, int64_t> hash_counts;
  std::map<int64_t, std::set<uint64_t>, std::less<int64_t>> hash_counts_map;
  std::unordered_map<uint64_t, std::vector<int64_t>> hash_op_map;

private:
  LgCache() = default;
  ~LgCache() = default;
  LgCache(const LgCache &) = delete;
  LgCache &operator=(const LgCache &) = delete;
};

} // namespace tpu
} // namespace tpu_mlir

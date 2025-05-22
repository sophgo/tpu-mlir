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
#include "tpu_mlir/Support/Module.h"
#include <filesystem>
#include <functional>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/WithColor.h>
#include <map>
#include <unordered_set>
#include <vector>

namespace tpu_mlir {
namespace tpu {

struct PairHash {
  size_t operator()(const std::pair<int64_t, int64_t> &p) const {
    return std::hash<int64_t>()(p.first) ^
           (std::hash<int64_t>()(p.second) << 1);
  }
};

typedef enum {
  DEBUGGER_DUMP_ALL = 0,   /* dump debug infos for all groups */
  DEBUGGER_DUMP_GIVEN = 1, /* dump debug infos only for given groups in
                              'conditional_debug_groups' */
  DEBUGGER_DUMP_MIXED =
      2, /* dump debug infos with type in 'conditional_debug_types'
            only for given groups and dump other debug infos for all groups */
  DEBUGGER_TRY_GIVEN = 3,
  DEBUGGER_DO_NOTHING
} lg_debugger_type_t;

typedef enum { SC_QUICK_SEARCH = 0, SC_BRUTE_FORCE = 1 } lg_sc_method_t;

extern std::map<lg_debugger_type_t, std::string> lg_debugger_type_str_map;
extern std::map<std::string, lg_debugger_type_t> lg_debugger_str_type_map;
extern std::map<lg_sc_method_t, std::string> lg_sc_method_type_str_map;
extern std::map<std::string, lg_sc_method_t> lg_sc_method_str_type_map;

struct LgDebuggerInfo {
  LgDebuggerInfo() { this->clear(); }
  ~LgDebuggerInfo() { this->clear(); }
  void clear() {
    this->func_start_idx = -1;
    this->func_end_idx = -1;
  }

  // debug group interval
  int64_t func_start_idx = -1;
  int64_t func_end_idx = -1;
};

class LgDebugger {
public:
  LgDebugger(const LgDebugger &) = delete;
  LgDebugger &operator=(const LgDebugger &) = delete;

  static LgDebugger &getInstance() {
    static LgDebugger instance;
    return instance;
  }

  void clear() {
    type_ = DEBUGGER_DO_NOTHING;
    sc_method_ = SC_QUICK_SEARCH;
    do_debug_ = false;
    conditional_debug_types_.clear();
    conditional_debug_groups_.clear();
    lg_debugger_infos_.clear();
  }

  void set_type(lg_debugger_type_t type) { this->type_ = type; }

  void set_type(std::string type) {
    if (lg_debugger_str_type_map.find(type) != lg_debugger_str_type_map.end()) {
      this->type_ = lg_debugger_str_type_map[type];
    } else {
      llvm_unreachable("unknown lg_debugger_type_t");
    }
  }

  lg_debugger_type_t get_type() { return type_; }

  std::string get_type_str() { return lg_debugger_type_str_map[type_]; }

  void set_sc_method(lg_sc_method_t method) { this->sc_method_ = method; }

  void set_sc_method(std::string method) {
    if (lg_sc_method_str_type_map.find(method) !=
        lg_sc_method_str_type_map.end()) {
      this->sc_method_ = lg_sc_method_str_type_map[method];
    } else {
      llvm_unreachable("unknown lg_sc_method_t");
    }
  }

  lg_sc_method_t get_sc_method() { return sc_method_; }

  std::string get_sc_method_str() {
    return lg_sc_method_type_str_map[sc_method_];
  }

  void set_do_debug(bool do_debug) { this->do_debug_ = do_debug; }

  bool get_do_debug() { return do_debug_; }

  void set_show_function_location(bool show_function_location) {
    this->show_function_location_ = show_function_location;
  }

  bool get_show_function_location() { return show_function_location_; }

  void add_conditional_debug_type(std::string type) {
    conditional_debug_types_.insert(type);
  }

  void add_conditional_debug_group(int64_t start, int64_t end) {
    conditional_debug_groups_.insert(std::make_pair(start, end));
  }

  std::unordered_set<std::pair<int64_t, int64_t>, PairHash>
  get_conditional_debug_groups() {
    return conditional_debug_groups_;
  }

  bool is_conditional_debug_group(int64_t start, int64_t end) {
    return conditional_debug_groups_.find(std::make_pair(start, end)) !=
           conditional_debug_groups_.end();
  }

  std::unordered_set<std::string> get_conditional_debug_types() {
    return conditional_debug_types_;
  }

  bool is_conditional_debug_type(std::string type) {
    return conditional_debug_types_.find(type) !=
           conditional_debug_types_.end();
  }

  void add_lg_debugger_info(int64_t start, int64_t end) {
    LgDebuggerInfo info;
    info.func_start_idx = start;
    info.func_end_idx = end;
    lg_debugger_infos_.push_back(info);
  }

  template <typename F>
  void debug_with_type(const char *debug_type, const char *file,
                       const int64_t line, const char *func, F &&callback) {
    if (!do_debug_)
      return;

    std::string range = "all";

    DEBUG_WITH_TYPE(debug_type, {
      llvm::dbgs() << LOG_ACTION(debug_type) << LOG_RANGE(range);

      if (show_function_location_) {
        llvm::dbgs() << LOG_LOC(file, line, func);
      }

      callback();
    });
  }

  template <typename F>
  void debug_with_type(const char *debug_type, const LgInfo &lg_info,
                       const char *file, const int64_t line, const char *func,
                       F &&callback) {
    if (!do_debug_)
      return;

    const auto debug_group =
        std::make_pair(lg_info.func_start_idx, lg_info.func_end_idx);
    const bool find_debug_group = conditional_debug_groups_.count(debug_group);
    const bool find_debug_type = conditional_debug_types_.count(debug_type);

    // determine whether to dump based on the debug_type and the debug group
    auto should_dump = [&]() {
      switch (type_) {
      case DEBUGGER_DUMP_ALL:
        return true;
      case DEBUGGER_DUMP_GIVEN:
        return find_debug_group;
      case DEBUGGER_DUMP_MIXED:
        return find_debug_type ? find_debug_group : true;
      default:
        return false;
      }
    }();

    if (!should_dump)
      return;

    // set range based on the type and whether the debug group is found
    auto get_range_str = [&]() {
      switch (type_) {
      case DEBUGGER_DUMP_ALL:
        return "all";
      case DEBUGGER_DUMP_GIVEN:
        return "given";
      case DEBUGGER_DUMP_MIXED:
        return find_debug_group ? "given" : "all";
      default:
        return "all";
      }
    }();
    std::string range = get_range_str;

    // log the debug information
    DEBUG_WITH_TYPE(debug_type, {
      llvm::dbgs() << LOG_ACTION(debug_type) << LOG_RANGE(range);

      if (show_function_location_) {
        llvm::dbgs() << LOG_LOC(file, line, func);
      }

      callback();
    });
  }

  void dump();
  void create_debugger_config(const std::string &config_file);
  void load_debugger_config(const std::string &config_file);

private:
  LgDebugger() { clear(); }
  ~LgDebugger() { clear(); }

  lg_debugger_type_t type_ = DEBUGGER_DO_NOTHING;
  lg_sc_method_t sc_method_ = SC_QUICK_SEARCH;
  bool do_debug_ = false;               // do debug or not
  bool show_function_location_ = false; // show function location or not
  std::unordered_set<std::string> conditional_debug_types_;
  std::unordered_set<std::pair<int64_t, int64_t>, PairHash>
      conditional_debug_groups_;
  std::vector<LgDebuggerInfo> lg_debugger_infos_;
};

} // namespace tpu
} // namespace tpu_mlir

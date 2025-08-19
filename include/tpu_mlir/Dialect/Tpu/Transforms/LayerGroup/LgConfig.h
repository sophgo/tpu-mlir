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
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tpu_mlir {
namespace tpu {

using ConfigValue = std::variant<bool, int, long, float, std::string>;
using ConfigMap = std::map<std::string, ConfigValue>;
struct ConfigEntry {
  std::string dtype;
  ConfigValue value;
};

class LgConfig {
public:
  LgConfig(const LgConfig &) = delete;
  LgConfig &operator=(const LgConfig &) = delete;

  static LgConfig &getInstance() {
    static LgConfig instance;
    return instance;
  }

  void clear() { sc_method_configs_.clear(); }

  void set_shape_secs_search_strategy(int strategy) {
    shape_secs_search_strategy_ =
        static_cast<shape_secs_search_strategy_t>(strategy);
  }

  shape_secs_search_strategy_t get_shape_secs_search_strategy() const {
    return shape_secs_search_strategy_;
  }

  void
  set_sc_method_configs(std::map<std::string, ConfigMap> sc_method_configs) {
    sc_method_configs_ = sc_method_configs;
  }

  std::map<std::string, ConfigMap> get_sc_method_configs() const {
    return sc_method_configs_;
  }

  template <typename T>
  T get_config_value(const std::string &sc_method,
                     const std::string &config_name, const T &default_value) {
    auto method_it = sc_method_configs_.find(sc_method);
    if (method_it == sc_method_configs_.end()) {
      return default_value;
    }

    auto config_it = method_it->second.find(config_name);
    if (config_it == method_it->second.end()) {
      return default_value;
    }

    return std::visit(
        [&](auto &&arg) -> T {
          using U = std::decay_t<decltype(arg)>;

          if constexpr (std::is_same_v<U, T>) {
            return arg;
          } else if constexpr (std::is_arithmetic_v<T> &&
                               std::is_arithmetic_v<U>) {
            return static_cast<T>(arg);
          } else if constexpr (std::is_same_v<T, std::string>) {
            if constexpr (std::is_arithmetic_v<U>) {
              return std::to_string(arg);
            } else if constexpr (std::is_same_v<U, bool>) {
              return arg ? "true" : "false";
            }
          } else if constexpr (std::is_same_v<T, bool>) {
            if constexpr (std::is_arithmetic_v<U>) {
              return arg != 0;
            } else if constexpr (std::is_same_v<U, std::string>) {
              return (arg == "true") || (arg == "1");
            }
          }
          return default_value;
        },
        config_it->second);
  }

  void load(const std::string &config_file);
  void dump();

private:
  LgConfig() { clear(); }
  ~LgConfig() { clear(); }

  shape_secs_search_strategy_t shape_secs_search_strategy_;
  std::map<std::string, ConfigMap> sc_method_configs_;
};

} // namespace tpu
} // namespace tpu_mlir

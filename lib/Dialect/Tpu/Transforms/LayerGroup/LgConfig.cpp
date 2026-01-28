//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgConfig.h"

namespace tpu_mlir {
namespace tpu {

void LgConfig::dump() {
  llvm::outs() << "Dumping LgConfig!\n";
  llvm::outs() << "Global Configs:\n";
  for (const auto &global_config : global_configs_) {
    llvm::outs() << "  " << global_config.first << " = ";
    std::visit([](auto &&arg) { llvm::outs() << arg << "\n"; },
               global_config.second);
  }
  for (const auto &sc_method : sc_method_configs_) {
    llvm::outs() << "Search Method Config: " << sc_method.first << "\n";
    for (const auto &config : sc_method.second) {
      llvm::outs() << "  " << config.first << " = ";
      std::visit([](auto &&arg) { llvm::outs() << arg << "\n"; },
                 config.second);
    }
  }
}

void LgConfig::load(const std::string &config_file) {
  // Load the debugger configuration json file
  auto bufferOrErr = llvm::MemoryBuffer::getFile(config_file);
  if (!bufferOrErr) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({
      llvm::errs() << "Can not open config file: "
                   << bufferOrErr.getError().message()
                   << ", using default layer group config!\n";
    });
    return;
  }
  // Parse JSON
  auto jsonOrErr = json::parse((*bufferOrErr)->getBuffer());
  if (!jsonOrErr) {
    llvm::errs() << "Failed to parse JSON: " << toString(jsonOrErr.takeError())
                 << "\n";
    return;
  }
  auto &root = *jsonOrErr;
  if (auto *rootObj = root.getAsObject()) {
    // set global_configs_
    if (auto shape_secs_search_strategy_int =
            rootObj->getInteger("shape_secs_search_strategy")) {
      global_configs_["shape_secs_search_strategy"] =
          static_cast<int>(*shape_secs_search_strategy_int);
    }
    if (auto structure_detect_opt_bool =
            rootObj->getBoolean("structure_detect_opt")) {
      global_configs_["structure_detect_opt"] = *structure_detect_opt_bool;
    }
    // set sc_method_configs
    if (auto sc_method_configs = rootObj->getArray("sc_method_configs")) {
      std::map<std::string, ConfigMap> configs_;
      for (const auto &config : *sc_method_configs) {
        if (auto config_obj = config.getAsObject()) {
          // set sc_method_
          std::string sc_method_str = "SC_QUICK_SEARCH";
          if (auto sc_method_s = config_obj->getString("sc_method")) {
            sc_method_str = sc_method_s->str();
          } else {
            llvm::dbgs()
                << "config sc_method not found, set to \'SC_QUICK_SEARCH\'\n";
          }

          // config list
          std::vector<std::string> config_list = {
              "MAX_TRY_NUM",
              "MAX_NSECS",
              "MAX_CSECS",
              "MAX_DSECS",
              "MAX_HSECS",
              "MAX_WSECS",
              "NSECS_SEARCH_RECORD_THRESHOLD",
              "CSECS_SEARCH_RECORD_THRESHOLD",
              "DSECS_SEARCH_RECORD_THRESHOLD",
              "HSECS_SEARCH_RECORD_THRESHOLD",
              "WSECS_SEARCH_RECORD_THRESHOLD"};

          // config dtype & default value
          std::map<std::string, ConfigEntry> config_init(
              {{"MAX_TRY_NUM", {"int", 20}},
               {"MAX_NSECS", {"int64_t", 32}},
               {"MAX_CSECS", {"int64_t", 32}},
               {"MAX_DSECS", {"int64_t", 32}},
               {"MAX_HSECS", {"int64_t", 32}},
               {"MAX_WSECS", {"int64_t", 32}},
               {"NSECS_SEARCH_RECORD_THRESHOLD", {"int", -1}},
               {"CSECS_SEARCH_RECORD_THRESHOLD", {"int", -1}},
               {"DSECS_SEARCH_RECORD_THRESHOLD", {"int", -1}},
               {"HSECS_SEARCH_RECORD_THRESHOLD", {"int", -1}},
               {"WSECS_SEARCH_RECORD_THRESHOLD", {"int", -1}}});

          for (auto config_name : config_list) {
            if (auto it = config_init.find(config_name);
                it != config_init.end()) {
              auto dtype = it->second.dtype;
              auto value = it->second.value;
              if (dtype == "int") {
                if (auto config_int = config_obj->getInteger(config_name)) {
                  configs_[sc_method_str][config_name] =
                      static_cast<int>(*config_int);
                } else {
                  configs_[sc_method_str][config_name] = value;
                }
              } else if (dtype == "int64_t") {
                if (auto config_int64 = config_obj->getInteger(config_name)) {
                  configs_[sc_method_str][config_name] =
                      static_cast<long>(*config_int64);
                } else {
                  configs_[sc_method_str][config_name] = value;
                }
              } else if (dtype == "float") {
                if (auto config_float = config_obj->getNumber(config_name)) {
                  configs_[sc_method_str][config_name] =
                      static_cast<float>(*config_float);
                } else {
                  configs_[sc_method_str][config_name] = value;
                }
              } else if (dtype == "bool") {
                if (auto config_bool = config_obj->getBoolean(config_name)) {
                  configs_[sc_method_str][config_name] = *config_bool;
                } else {
                  configs_[sc_method_str][config_name] = value;
                }
              } else if (dtype == "string") {
                if (auto config_string = config_obj->getString(config_name)) {
                  configs_[sc_method_str][config_name] = config_string->str();
                } else {
                  configs_[sc_method_str][config_name] = value;
                }
              } else {
                llvm_unreachable("unknown config dtype");
              }
            }
          }
        }
      }
      set_sc_method_configs(configs_);
    }

    dump();
    llvm::WithColor(llvm::outs(), llvm::raw_ostream::GREEN) << llvm::format(
        "Load debugger file \"%s\" success!\n", config_file.c_str());
  }
}

uint64_t LgConfig::get_config_hash() const {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);

  // os << "Strategy Using: " << shape_secs_search_strategy_ << "\n";
  for (const auto &sc_method : sc_method_configs_) {
    os << "Search Method Config: " << sc_method.first << "\n";
    for (const auto &config : sc_method.second) {
      os << "  Config Name: " << config.first << "\n";
      os << "    Value: ";
      std::visit([&os](auto &&arg) { os << arg << "\n"; }, config.second);
    }
  }

  os.flush();

  return llvm::xxh3_64bits(llvm::StringRef(buffer.data(), buffer.size()));
}

} // namespace tpu
} // namespace tpu_mlir

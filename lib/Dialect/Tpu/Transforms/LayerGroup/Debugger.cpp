//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/Debugger.h"

namespace tpu_mlir {
namespace tpu {

std::map<lg_debugger_type_t, std::string> lg_debugger_type_str_map = {
    {DEBUGGER_DUMP_ALL, "DEBUGGER_DUMP_ALL"},
    {DEBUGGER_DUMP_GIVEN, "DEBUGGER_DUMP_GIVEN"},
    {DEBUGGER_DUMP_MIXED, "DEBUGGER_DUMP_MIXED"},
    {DEBUGGER_TRY_GIVEN, "DEBUGGER_TRY_GIVEN"},
    {DEBUGGER_DO_NOTHING, "DEBUGGER_DO_NOTHING"}};

std::map<std::string, lg_debugger_type_t> lg_debugger_str_type_map = {
    {"DEBUGGER_DUMP_ALL", DEBUGGER_DUMP_ALL},
    {"DEBUGGER_DUMP_GIVEN", DEBUGGER_DUMP_GIVEN},
    {"DEBUGGER_DUMP_MIXED", DEBUGGER_DUMP_MIXED},
    {"DEBUGGER_TRY_GIVEN", DEBUGGER_TRY_GIVEN},
    {"DEBUGGER_DO_NOTHING", DEBUGGER_DO_NOTHING}};

std::map<lg_sc_method_t, std::string> lg_sc_method_type_str_map = {
    {SC_QUICK_SEARCH, "SC_QUICK_SEARCH"}, {SC_BRUTE_FORCE, "SC_BRUTE_FORCE"}};

std::map<std::string, lg_sc_method_t> lg_sc_method_str_type_map = {
    {"SC_QUICK_SEARCH", SC_QUICK_SEARCH}, {"SC_BRUTE_FORCE", SC_BRUTE_FORCE}};

void LgDebugger::dump() {
  // dump type_
  llvm::dbgs() << LOG_ACTION("debugger") << LOG_STEP("type_")
               << LOG_KV("type_", get_type_str()) << "\n";

  // dump conditional_debug_types_
  llvm::dbgs() << LOG_ACTION("debugger") << LOG_STEP("conditional_debug_types_")
               << "; conditional_debug_types_ = ";
  const char *sep = "\"";
  for (auto debug_type : conditional_debug_types_) {
    llvm::dbgs() << sep << debug_type;
    sep = "\", \"";
  }
  llvm::dbgs() << "\"\n";

  // dump conditional_debug_groups_
  int i = 1;
  for (const auto &group : conditional_debug_groups_) {
    llvm::dbgs() << LOG_ACTION("debugger")
                 << LOG_STEP("conditional_debug_groups_")
                 << LOG_KV("dump_idx", i++)
                 << LOG_KV("func_start_idx", group.first)
                 << LOG_KV("func_end_idx", group.second) << "\n";
  }

  // dump lg_debugger_infos_
  i = 1;
  for (const auto &info : lg_debugger_infos_) {
    llvm::dbgs() << LOG_ACTION("debugger") << LOG_STEP("lg_debugger_infos_")
                 << LOG_KV("dump_idx", i++)
                 << LOG_KV("func_start_idx", info.func_start_idx)
                 << LOG_KV("func_end_idx", info.func_end_idx) << "\n";
  }
}

void LgDebugger::create_debugger_config(const std::string &config_file) {
  llvm::outs() << "Try to create debugger file\n";

  if (std::filesystem::exists(config_file)) {
    llvm::WithColor(llvm::outs(), llvm::raw_ostream::YELLOW) << llvm::format(
        "Debugger file \"%s\" already exists! Just do nothing.\n",
        (const char *)config_file.c_str());
    return;
  }

  std::error_code EC;
  llvm::raw_fd_ostream OS(config_file, EC);
  if (EC) {
    llvm::errs() << "Failed to open file for writing: " << EC.message() << "\n";
    return;
  }

  json::OStream J(OS, 2);
  J.objectBegin();

  // show debugger type options
  J.attributeBegin("type options");
  J.arrayBegin();
  for (const auto &type : lg_debugger_type_str_map) {
    J.value(type.second);
  }
  J.arrayEnd();
  J.attributeEnd();

  // show sc_method options
  J.attributeBegin("sc_method options");
  J.arrayBegin();
  for (const auto &method : lg_sc_method_type_str_map) {
    J.value(method.second);
  }
  J.arrayEnd();
  J.attributeEnd();

  // init debugger type
  J.attribute("type", get_type_str());

  // init sc_method
  J.attribute("sc_method", get_sc_method_str());

  // init list of debug type which will be dumped conditionally
  // init with empty array
  J.attributeBegin("conditional_debug_types");
  J.arrayBegin();
  J.arrayEnd();
  J.attributeEnd();

  // init whether show function location
  J.attribute("show_function_location", false);

  // write GroupLayer array as template
  J.attributeBegin("GroupLayerDump");
  J.arrayBegin();
  J.objectBegin();
  J.attribute("func_start_idx", -1);
  J.attribute("func_end_idx", -1);
  J.objectEnd();
  J.arrayEnd();
  J.attributeEnd();

  J.objectEnd();

  if (std::filesystem::exists(config_file)) {
    llvm::WithColor(llvm::outs(), llvm::raw_ostream::GREEN)
        << llvm::format("Create debugger file \"%s\"success!\n",
                        (const char *)config_file.c_str());
  } else {
    llvm::WithColor(llvm::outs(), llvm::raw_ostream::RED)
        << llvm::format("Create debugger file \"%s\" failed!\n",
                        (const char *)config_file.c_str());
  }
}

void LgDebugger::load_debugger_config(const std::string &config_file) {
  // Load the debugger configuration json file
  auto bufferOrErr = llvm::MemoryBuffer::getFile(config_file);
  if (!bufferOrErr) {
    llvm::errs() << "Failed to open file: " << bufferOrErr.getError().message()
                 << "\n";
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
    // set type_
    if (auto type_s = rootObj->getString("type")) {
      set_type(type_s->str());
    } else {
      llvm::dbgs() << "debugger type not found, set to \'DEBUGGER_DUMP_ALL\'\n";
    }
    // set sc_method_
    if (auto sc_method_s = rootObj->getString("sc_method")) {
      set_sc_method(sc_method_s->str());
    } else {
      llvm::dbgs()
          << "debugger sc_method not found, set to \'SC_QUICK_SEARCH\'\n";
    }
    // set show_function_location_ boolean
    if (auto show_function_location =
            rootObj->getBoolean("show_function_location")) {
      show_function_location_ = *show_function_location;
    } else {
      llvm::dbgs()
          << "debugger show_function_location not found, set to \'false\'\n";
    }
    // set conditional_debug_types_
    if (auto conditional_debug_types_ =
            rootObj->getArray("conditional_debug_types")) {
      for (const auto debug_type : *conditional_debug_types_) {
        if (auto debug_type_s = debug_type.getAsString()) {
          add_conditional_debug_type(debug_type_s->str());
        } else {
          llvm_unreachable("debugger type needs to be string");
        }
      }
    }
    // set conditional_debug_groups_ & lg_debugger_infos_
    if (type_ == DEBUGGER_DUMP_GIVEN || type_ == DEBUGGER_DUMP_MIXED) {
      if (auto groupLayerArray = rootObj->getArray("GroupLayerDump")) {
        for (const auto &groupObj : *groupLayerArray) {
          int64_t _func_start_idx = -1, _func_end_idx = -1;
          if (auto *groupObj_ = groupObj.getAsObject()) {
            if (auto func_start_idx = groupObj_->getInteger("func_start_idx")) {
              _func_start_idx = *func_start_idx;
            } else {
              llvm_unreachable("func_start_idx needs to be int");
            }
            if (auto func_end_idx = groupObj_->getInteger("func_end_idx")) {
              _func_end_idx = *func_end_idx;
            } else {
              llvm_unreachable("func_end_idx needs to be int");
            }
          } else {
            llvm_unreachable("\"GroupLayerDump\" array format error");
          }
          add_lg_debugger_info(_func_start_idx, _func_end_idx);
          add_conditional_debug_group(_func_start_idx, _func_end_idx);
        }
      }
    }
    do_debug_ = true;

    debug_with_type("debugger", __FILE__, __LINE__, __FUNCTION__, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO("dump", "call_function",
                                            "dump debugger settings")
                   << "\n";
      dump();
    });
    llvm::WithColor(llvm::outs(), llvm::raw_ostream::GREEN) << llvm::format(
        "Load debugger file \"%s\" success!\n", config_file.c_str());
  }
}

} // namespace tpu
} // namespace tpu_mlir

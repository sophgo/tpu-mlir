//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <filesystem>
#include <llvm/Support/FormatAdapters.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/WithColor.h>
#include <map>
#include <set>
#include <variant>

namespace tpu_mlir {

/*
 * @file RewriteConfigUtils.h
 * @brief Utility functions for loading and parsing rewrite rules from JSON
 * config files.
 *
 * This file provides functions to load rewrite rules from a JSON configuration
 * file, parse parameters in a type-safe manner, and dump the loaded rules for
 * debugging purposes. It supports various parameter types including boolean,
 * integer, float, string, and vectors of these types. The rules are defined in
 * a structured format that allows for easy modification and extension. The
 * configuration file should follow a specific JSON schema to ensure
 * compatibility with the loading functions. Example JSON schema:
 * {
 *   "rewriter_rules": [
 *     {
 *       "pattern_name": "SamplePattern1",
 *       "params": {
 *         "attr_1": true,
 *         "attr_2": 42,
 *         "attr_3": "fused_add"
 *       }
 *     },
 *     {
 *       "pattern_name": "SamplePattern2",
 *       "params": {
 *         "attr1": 1.5,
 *        "attr2": [8, 256, 1024]
 *       }
 *     }
 *   ]
 * }
 */

using ConfigValue = std::variant<bool, int, float, std::string,
                                 std::vector<bool>, std::vector<int>,
                                 std::vector<float>, std::vector<std::string>>;
using ParamMap = std::map<std::string, ConfigValue>;

struct RewriterRule {
  std::string pattern_name;
  ParamMap params;
};

// load rewrite rules from JSON config file
std::vector<RewriterRule> loadRewriteConfig(const std::string &configPath);

// get scalar parameter from ParamMap with a default value
template <typename T>
T getParam(const ParamMap &params, llvm::StringRef key, T defaultValue);

// get vector parameter from ParamMap with a default value
template <typename T>
std::vector<T> getParamVector(const ParamMap &params, llvm::StringRef key,
                              std::vector<T> defaultValue = {});

// explicit template instantiation declarations (scalar types)
extern template int getParam<int>(const ParamMap &, llvm::StringRef, int);
extern template bool getParam<bool>(const ParamMap &, llvm::StringRef, bool);
extern template float getParam<float>(const ParamMap &, llvm::StringRef, float);
extern template std::string getParam<std::string>(const ParamMap &,
                                                  llvm::StringRef, std::string);

// explicit template instantiation declarations (vector types)
extern template std::vector<int>
getParamVector<int>(const ParamMap &, llvm::StringRef, std::vector<int>);
extern template std::vector<bool>
getParamVector<bool>(const ParamMap &, llvm::StringRef, std::vector<bool>);
extern template std::vector<float>
getParamVector<float>(const ParamMap &, llvm::StringRef, std::vector<float>);
extern template std::vector<std::string>
getParamVector<std::string>(const ParamMap &, llvm::StringRef,
                            std::vector<std::string>);

void dumpVariant(const ConfigValue &value, llvm::raw_ostream &os);
void dumpRewriterRules(const std::vector<RewriterRule> &rules,
                       llvm::raw_ostream &os = llvm::outs(),
                       bool verbose = false);
void dumpRewriterRule(const RewriterRule &rule,
                      llvm::raw_ostream &os = llvm::outs());

} // namespace tpu_mlir

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/RewriterConfigUtils.h"
#include "tpu_mlir/Support/Logger.h"

namespace tpu_mlir {

ParamMap parseParamMap(const llvm::json::Object *paramsObj) {
  ParamMap params;
  if (!paramsObj)
    return params;

  for (const auto &[key, value] : *paramsObj) {
    std::string key_str = key.str();
    // 1. process boolean
    if (auto boolVal = value.getAsBoolean()) {
      params[key_str] = *boolVal;
    }
    // 2. process integer
    else if (auto intVal = value.getAsInteger()) {
      params[key_str] = static_cast<int>(*intVal);
    }
    // 3. process float
    else if (auto doubleVal = value.getAsNumber()) {
      params[key_str] = static_cast<float>(*doubleVal);
    }
    // 4. process string
    else if (auto strVal = value.getAsString()) {
      params[key_str] = strVal->str();
    }
    // 5. process array
    else if (auto arrayVal = value.getAsArray()) {
      // try process as array of bool
      if (!arrayVal->empty() && (*arrayVal)[0].getAsBoolean()) {
        std::vector<bool> boolVec;
        for (const auto &elem : *arrayVal) {
          if (auto b = elem.getAsBoolean()) {
            boolVec.push_back(*b);
          } else {
            llvm::errs() << "Warning: Mixed types in boolean array '" << key_str
                         << "'. Skipping.\n";
            boolVec.clear();
            break;
          }
        }
        if (!boolVec.empty()) {
          params[key_str] = boolVec;
          continue;
        }
      }

      // try process as array of integers
      if (!arrayVal->empty() && (*arrayVal)[0].getAsInteger()) {
        std::vector<int> intVec;
        for (const auto &elem : *arrayVal) {
          if (auto i = elem.getAsInteger()) {
            intVec.push_back(static_cast<int>(*i));
          } else {
            llvm::errs() << "Warning: Mixed types in integer array '" << key_str
                         << "'. Skipping.\n";
            intVec.clear();
            break;
          }
        }
        if (!intVec.empty()) {
          params[key_str] = intVec;
          continue;
        }
      }

      // try process as array of floats
      if (!arrayVal->empty() && (*arrayVal)[0].getAsNumber()) {
        std::vector<float> floatVec;
        for (const auto &elem : *arrayVal) {
          if (auto d = elem.getAsNumber()) {
            floatVec.push_back(static_cast<float>(*d));
          } else {
            llvm::errs() << "Warning: Mixed types in float array '" << key_str
                         << "'. Skipping.\n";
            floatVec.clear();
            break;
          }
        }
        if (!floatVec.empty()) {
          params[key_str] = floatVec;
          continue;
        }
      }

      // try process as array of strings
      if (!arrayVal->empty() && (*arrayVal)[0].getAsString()) {
        std::vector<std::string> strVec;
        for (const auto &elem : *arrayVal) {
          if (auto s = elem.getAsString()) {
            strVec.push_back(s->str());
          } else {
            llvm::errs() << "Warning: Mixed types in string array '" << key_str
                         << "'. Skipping.\n";
            strVec.clear();
            break;
          }
        }
        if (!strVec.empty()) {
          params[key_str] = strVec;
          continue;
        }
      }

      // if we reach here, it means the array is empty or unsupported type
      if (arrayVal->empty()) {
        llvm::errs() << "Warning: Empty array for parameter '" << key_str
                     << "'. Using empty float vector.\n";
        params[key_str] = std::vector<float>{};
      } else {
        llvm::errs()
            << "Warning: Unsupported array type for parameter '" << key_str
            << "'. Supported types: bool, integer, float, string arrays.\n";
      }
    } else {
      llvm::errs() << "Warning: Unsupported JSON type for parameter '"
                   << key_str
                   << "'. Supported types: bool, integer, float, string.\n";
    }
  }
  return params;
}

std::vector<RewriterRule> loadRewriteConfig(const std::string &configPath) {
  // Check if file exists
  if (std::filesystem::exists(configPath)) {
    PASS_LOG_DEBUG_BLOCK({
      llvm::WithColor(llvm::outs(), llvm::raw_ostream::GREEN)
          << "Configuration file has found: " << configPath << "\n";
    });
  } else {
    return {};
  }

  // Read file content
  auto fileOrErr = llvm::MemoryBuffer::getFile(configPath);
  if (auto err = fileOrErr.getError()) {
    llvm::errs() << "Error reading config file: " << err.message() << "\n";
    return {};
  }

  // Parse JSON
  llvm::Expected<llvm::json::Value> jsonOrErr =
      llvm::json::parse(fileOrErr.get()->getBuffer());
  if (auto err = jsonOrErr.takeError()) {
    llvm::errs() << "JSON parse error: " << llvm::toString(std::move(err))
                 << "\n";
    return {};
  }

  // Extract top-level object
  llvm::json::Object *root = jsonOrErr->getAsObject();
  if (!root) {
    llvm::errs() << "Error: Configuration root is not a JSON object\n";
    return {};
  }

  // Extract rules array
  llvm::json::Array *rulesArray = root->getArray("rewriter_rules");
  if (!rulesArray) {
    llvm::errs() << "Error: Missing 'rewriter_rules' array in configuration\n";
    return {};
  }

  // Parse each rule
  std::vector<RewriterRule> rules;
  for (const llvm::json::Value &ruleVal : *rulesArray) {
    if (auto *ruleObj = ruleVal.getAsObject()) {
      RewriterRule rule;

      // Parse operation type
      if (auto pattern_name = ruleObj->getString("pattern_name")) {
        rule.pattern_name = pattern_name->str();
      } else {
        llvm::errs() << "Warning: Rule missing 'pattern_name', skipping\n";
        continue;
      }

      // Parse parameters
      if (auto *paramsObj = ruleObj->getObject("params")) {
        rule.params = parseParamMap(paramsObj);
      }

      rules.push_back(std::move(rule));
    }
  }

  if (rules.empty()) {
    llvm::errs() << "Warning: No valid rules found in configuration\n";
  }

  return rules;
}

// Template implementation for getParam
template <typename T>
T getParam(const ParamMap &params, llvm::StringRef key, T defaultValue) {
  auto it = params.find(key.str());
  if (it == params.end())
    return defaultValue;

  try {
    return std::get<T>(it->second);
  } catch (const std::bad_variant_access &) {
    llvm::errs() << "Warning: Type mismatch for parameter '" << key
                 << "'. Using default value.\n";
    return defaultValue;
  }
}

// Template implementation for getParamVector
template <typename T>
std::vector<T> getParamVector(const ParamMap &params, llvm::StringRef key,
                              std::vector<T> defaultValue) {
  auto it = params.find(key.str());
  if (it == params.end())
    return defaultValue;

  if (auto *vec = std::get_if<std::vector<T>>(&it->second)) {
    // if it is already a vector<T>, return it
    return *vec;
  } else if (auto *scalar = std::get_if<T>(&it->second)) {
    // if it is a scalar T, convert it to vector<T>
    llvm::errs() << "Warning: Scalar value for vector parameter '" << key
                 << "'. Converting to vector.\n";
    return std::vector<T>{*scalar};
  } else {
    llvm::errs() << "Warning: Type mismatch for vector parameter '" << key
                 << "'. Using default value.\n";
    return defaultValue;
  }
}

// print scalar
template <typename T>
void dumpScalar(const T &value, llvm::raw_ostream &os) {
  if constexpr (std::is_same_v<T, bool>) {
    os << (value ? "true" : "false");
  } else if constexpr (std::is_same_v<T, int>) {
    os << llvm::formatv("{0,-8}", value);
  } else if constexpr (std::is_same_v<T, float>) {
    os << llvm::formatv("{0:F2}", value);
  } else if constexpr (std::is_same_v<T, std::string>) {
    if (value.length() > 15) {
      os << "\"" << value.substr(0, 12) << "...\"";
    } else {
      os << "\"" << value << "\"";
    }
  } else {
    os << value;
  }
}

// print vector
template <typename T>
void dumpVector(const std::vector<T> &vec, llvm::raw_ostream &os) {
  os << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    dumpScalar(vec[i], os);
    if (i < vec.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
}

void dumpVariant(const ConfigValue &value, llvm::raw_ostream &os) {
  std::visit(
      [&](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;

        if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, int> ||
                      std::is_same_v<T, float> || std::is_same_v<T, double> ||
                      std::is_same_v<T, std::string>) {
          // scalar type
          dumpScalar(arg, os);
        } else if constexpr (std::is_same_v<T, std::vector<bool>> ||
                             std::is_same_v<T, std::vector<int>> ||
                             std::is_same_v<T, std::vector<float>> ||
                             std::is_same_v<T, std::vector<double>> ||
                             std::is_same_v<T, std::vector<std::string>>) {
          // array type
          dumpVector(arg, os);
        } else {
          // unsupported type
          os << "Unsupported type";
        }
      },
      value);
}

std::string paramMapToString(const ParamMap &params) {
  std::string result;
  llvm::raw_string_ostream oss(result);

  oss << "{";
  bool first = true;
  for (const auto &[key, value] : params) {
    if (!first)
      oss << ", ";
    first = false;

    oss << key << ": ";
    dumpVariant(value, oss);
  }

  oss << "}";
  return oss.str();
}

void dumpRewriterRules(const std::vector<RewriterRule> &rules,
                       llvm::raw_ostream &os, bool verbose) {
  if (rules.empty()) {
    os << "No rewrite rules loaded.\n";
    return;
  }

  // table header
  os << "Rewrite Rules Configuration (" << rules.size() << " rules):\n";
  os << "==========================================================\n";

  int maxIndexWidth = 5;        // Index width
  int maxPatternNameWidth = 25; // Pattern Name width
  int maxParamWidth = 22;       // Parameter width
  int maxKeyWidth = 22;         // Key width

  // simple mode
  if (!verbose) {
    std::map<std::string, std::set<std::string>> pattern_groups;

    for (const auto &rule : rules) {
      auto params_str = paramMapToString(rule.params);
      pattern_groups[rule.pattern_name].insert(params_str);
    }

    os << llvm::formatv(
        "{0} | {1} | {2}\n",
        llvm::fmt_align("Index", llvm::AlignStyle::Left, maxIndexWidth),
        llvm::fmt_align("Pattern Name", llvm::AlignStyle::Left,
                        maxPatternNameWidth),
        llvm::fmt_align("Parameter Count", llvm::AlignStyle::Left,
                        maxParamWidth));
    os << llvm::formatv("{0} | {1} | {2}\n",
                        llvm::fmt_repeat("-", maxIndexWidth),
                        llvm::fmt_repeat("-", maxPatternNameWidth),
                        llvm::fmt_repeat("-", maxParamWidth));
    int index = 0;
    for (const auto &[pattern_name, params_set] : pattern_groups) {
      os << llvm::formatv(
          "{0} | {1} | {2}\n",
          llvm::fmt_align(index++, llvm::AlignStyle::Left, maxIndexWidth),
          llvm::fmt_align(pattern_name, llvm::AlignStyle::Left,
                          maxPatternNameWidth),
          llvm::fmt_align(params_set.size(), llvm::AlignStyle::Left,
                          maxParamWidth));
    }

    os << "==========================================================\n";
    os << "\nUse verbose mode for detailed parameter view.\n";
    return;
  }

  // verbose mode
  os << llvm::formatv(
      "{0} | {1} | {2}\n",
      llvm::fmt_align("Index", llvm::AlignStyle::Left, maxIndexWidth),
      llvm::fmt_align("Pattern Name", llvm::AlignStyle::Left,
                      maxPatternNameWidth),
      llvm::fmt_align("Parameter Information", llvm::AlignStyle::Left,
                      maxParamWidth));

  for (size_t i = 0; i < rules.size(); ++i) {
    const auto &rule = rules[i];
    os << llvm::formatv("{0} | {1} | {2}\n",
                        llvm::fmt_repeat("-", maxIndexWidth),
                        llvm::fmt_repeat("-", maxPatternNameWidth),
                        llvm::fmt_repeat("-", maxParamWidth));

    os << llvm::formatv(
        "{0} | {1} | ",
        llvm::fmt_align(i, llvm::AlignStyle::Left, maxIndexWidth),
        llvm::fmt_align(rule.pattern_name, llvm::AlignStyle::Left,
                        maxPatternNameWidth));

    if (rule.params.empty()) {
      os << "(no parameters)";
    } else {
      bool first = true;
      for (const auto &[key, value] : rule.params) {
        if (!first) {
          os << llvm::formatv(
              "\n{0} | {1} | ",
              llvm::fmt_align("", llvm::AlignStyle::Left, maxIndexWidth),
              llvm::fmt_align("", llvm::AlignStyle::Left, maxPatternNameWidth));
        }
        first = false;

        os << llvm::formatv("{0}", llvm::fmt_align(key, llvm::AlignStyle::Left,
                                                   maxKeyWidth))
           << ": ";
        dumpVariant(value, os);
      }
    }
    os << "\n";
  }

  os << "==========================================================\n";
}

void dumpRewriterRule(const RewriterRule &rule, llvm::raw_ostream &os) {
  int maxPatternNameWidth = 25; // Pattern Name width
  int maxParamWidth = 22;       // Parameter width
  int maxKeyWidth = 22;         // Key width

  os << "==========================================================\n";
  os << llvm::formatv("{0} | {1}\n",
                      llvm::fmt_align("Pattern Name", llvm::AlignStyle::Left,
                                      maxPatternNameWidth),
                      llvm::fmt_align("Parameter Information",
                                      llvm::AlignStyle::Left, maxParamWidth));
  os << llvm::formatv("{0} | {1}\n", llvm::fmt_repeat("-", maxPatternNameWidth),
                      llvm::fmt_repeat("-", maxParamWidth));

  os << llvm::formatv("{0} | ",
                      llvm::fmt_align(rule.pattern_name, llvm::AlignStyle::Left,
                                      maxPatternNameWidth));
  if (rule.params.empty()) {
    os << "(no parameters)";
  } else {
    bool first = true;
    for (const auto &[key, value] : rule.params) {
      if (!first) {
        os << llvm::formatv(
            "\n{0} | ",
            llvm::fmt_align("", llvm::AlignStyle::Left, maxPatternNameWidth));
      }
      first = false;

      os << llvm::formatv("{0}", llvm::fmt_align(key, llvm::AlignStyle::Left,
                                                 maxKeyWidth))
         << ": ";
      dumpVariant(value, os);
    }
  }
  os << "\n";
  os << "==========================================================\n";
}

// Explicit template instantiations
template int getParam<int>(const ParamMap &, llvm::StringRef, int);
template bool getParam<bool>(const ParamMap &, llvm::StringRef, bool);
template float getParam<float>(const ParamMap &, llvm::StringRef, float);
template std::string getParam<std::string>(const ParamMap &, llvm::StringRef,
                                           std::string);
template std::vector<int> getParamVector<int>(const ParamMap &, llvm::StringRef,
                                              std::vector<int>);
template std::vector<bool>
getParamVector<bool>(const ParamMap &, llvm::StringRef, std::vector<bool>);
template std::vector<float>
getParamVector<float>(const ParamMap &, llvm::StringRef, std::vector<float>);
template std::vector<std::string>
getParamVector<std::string>(const ParamMap &, llvm::StringRef,
                            std::vector<std::string>);

} // namespace tpu_mlir

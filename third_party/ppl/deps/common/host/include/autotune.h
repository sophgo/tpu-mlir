#ifndef __AUTOTUNE_H__
#define __AUTOTUNE_H__
#include "chip_map.h"
#include "host_def.h"
#include <any>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>
#ifdef PPL_TUNE
#include <chrono>
#include <sys/stat.h>
#include <tpuDNN.h>
#endif

namespace fs = std::filesystem;

#define MD5_LENS 10
#ifdef PPL_TUNE
using ParamValue = std::variant<int, unsigned long long, float, double, bool,
                                std::string, tpudnnHandle_t>;
using micro_double = std::chrono::duration<double, std::micro>;
#else
using ParamValue =
    std::variant<int, unsigned long long, float, double, bool, std::string>;
#endif
using TuneConfig = std::map<std::string, ParamValue>;
using Invariants = std::vector<std::string>;
using TunedFunction =
    std::function<int(const std::map<std::string, std::any> &)>;

struct TuneParam {
  std::vector<TuneConfig> configs;
  std::vector<std::string> tunable;
  std::vector<std::string> fixed;
  std::vector<std::string> full_param;
};

struct FunctionConfig {
  std::function<int(const std::vector<std::any> &)> func;
  TuneParam param;
  std::string md5;
  std::vector<int> rets;
  std::vector<double> times;
};

enum RuntimeType { RUNTIME_TPUV7, RUNTIME_BMLIB, RUNTIME_UNKNOW };

#ifdef PPL_TUNE
static std::vector<double>
run_autotune_script(const std::string &script_path,
                    const std::vector<std::string> &string_vector,
                    const std::string &chip, const std::string &mode,
                    bool verbose = false) {
  std::string command = "python3 " + script_path;

  for (const auto &arg : string_vector) {
    command += " \"" + arg + "\"";
  }
  command += " --chip \"" + chip + "\"";
  command += " --mode " + mode;
  if (verbose)
    command += " --verbose";
  std::array<char, 128> buffer;
  std::string result;
  // Executing command
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"),
                                                pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  // Get result
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  if (!result.empty()) {
    result.erase(result.find_last_not_of(" \n\r\t") + 1);
  }
  int status = pclose(pipe.release());
  if (status != 0) {
    std::string error_msg =
        "Python script execution failed with status: " + std::to_string(status);
    if (!result.empty()) {
      error_msg += "\nOutput: " + result;
    }
    throw std::runtime_error(error_msg);
  }
  std::stringstream ss(result);
  std::string line, last_line;
  while (std::getline(ss, line)) {
    if (!line.empty())
      last_line = line;
  }
  std::stringstream last_ss(last_line);
  std::vector<double> values;
  double val;
  while (last_ss >> val) {
    values.push_back(val);
  }
  return values;
}
#endif

class AutoTuner {
public:
  AutoTuner() {
    cache_path_ = std::string(getenv("HOME")) + "/.ppl/autotune/";
    if (auto env = getenv("AUTOTUNE_CACHE_PATH")) {
      cache_path_ = std::string(env);
    }
    mode_ = "pcie";
    if (auto env = getenv("PPL_TPUKERNEL_DEV_MODE")) {
      mode_ = std::string(env);
    }
    record_size_ = 4096;
    if (auto env = std::getenv("PROFILE_RECORD_SIZE")) {
      record_size_ = atoi(env);
    }
    iter_num_ = 3;
    if (auto env = std::getenv("PPL_AUTOTUNE_ITER_NUM")) {
      iter_num_ = atoi(env);
    }
    py_script_ = std::getenv("PPL_PROJECT_ROOT");
    py_script_ /= "python/tool/autotune.py";
    load_cache();
  }

#ifdef PPL_TUNE
  template <typename... Args>
  int profile(int (*func)(Args...), const std::string &chip_code,
              Args... args) {
    // wrap function call with lambda
    auto wraped_func = [func](const std::vector<std::any> &args) {
      std::tuple<Args...> tuple;
      fill_tuple_from_vector<0>(tuple, args);
      return std::apply(func, tuple);
    };
    int ret = call_func(wraped_func, {args...}, chip_code, "sim", 1);
    if (ret == 0) {
      std::string path = std::filesystem::current_path();
      run_autotune_script(py_script_, {path}, chip_code, mode_, true);
      std::cout << "Profile result stored in " + path << std::endl;
    }
    return ret;
  }

  template <typename... Args>
  void config(int (*func)(Args...), const std::string &func_name,
              const std::string &ir_md5, const std::vector<TuneConfig> &configs,
              const std::vector<std::string> &fixed,
              const std::vector<std::string> &full_param) {
    if (full_param.size() != sizeof...(Args)) {
      throw std::runtime_error("full_param size must match function arity");
    }
    FunctionConfig fc{};
    fc.param.full_param = full_param;
    fc.param.configs = configs;
    fc.md5 = ir_md5;
    fc.rets.resize(configs.size(), 0);
    fc.times.resize(configs.size(), 0.);

    // get tunable and make sure all configs has the same keys
    if (!configs.empty()) {
      auto &first_config = configs[0];
      std::set<std::string> tunable_keys;
      for (auto &[k, v] : first_config) {
        tunable_keys.insert(k);
      }
      for (auto &config : configs) {
        std::set<std::string> current_keys;
        for (auto &[k, v] : config) {
          current_keys.insert(k);
        }
        if (current_keys != tunable_keys) {
          throw std::runtime_error("All configs must have the same keys");
        }
      }
      // record as real order
      for (auto s : full_param) {
        if (std::count(fixed.begin(), fixed.end(), s) > 0)
          fc.param.fixed.push_back(s);
        else if (std::count(tunable_keys.begin(), tunable_keys.end(), s) > 0)
          fc.param.tunable.push_back(s);
      }
    }

    // wrap function call with lambda
    fc.func = [func](const std::vector<std::any> &args) {
      std::tuple<std::remove_reference_t<Args>...> tuple;
      fill_tuple_from_vector<0>(tuple, args);
      return std::apply(func, tuple);
    };

    functions_[func_name] = fc;
  }

  template <typename... Args>
  int tune(const std::string &func_name, const std::string &chip_code,
           Args... args) {
    // int tune(std::string func_name, Args... args) {
    if (functions_.find(func_name) == functions_.end()) {
      throw std::runtime_error("No function configured. Call config() first.");
    }
    auto &fc = functions_[func_name];
    std::vector<std::any> default_args = {args...};
    std::string key = func_name + "_" + fc.md5.substr(MD5_LENS) +
                      serialize_fixed(default_args, fc.param);
    fs::path path = cache_path_ / chip_code / key;
    if (fs::exists(path)) {
      fs::remove_all(path);
    }
    int ret;
    std::vector<double> times;
    // std::vector<std::string> profile_path;
    for (int i; i < fc.param.configs.size(); i++) {
      auto &config = fc.param.configs[i];
      std::string sub = "tmp";
      std::vector<std::any> current_args = default_args;
      // replace tunable parameters
      for (size_t i = 0; i < fc.param.tunable.size(); ++i) {
        auto it = std::find(fc.param.full_param.begin(),
                            fc.param.full_param.end(), fc.param.tunable[i]);
        if (it != fc.param.full_param.end()) {
          size_t pos = it - fc.param.full_param.begin();
          auto value = config.at(fc.param.tunable[i]);
          current_args[pos] = variant_to_any(value);
          sub += ("-" + fc.param.tunable[i] + "_" +
                  any_to_string(current_args[pos]));
        }
      }
      auto tmp_dir = path / sub;
      fs::create_directories(tmp_dir);
      chmod(tmp_dir.c_str(), 0777);
      fs::current_path(tmp_dir);
      // call function and measure time
      auto start = std::chrono::high_resolution_clock::now();
      ret = call_func(fc.func, current_args, chip_code, sub, iter_num_, false);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end - start;
      fc.rets[i] = ret;
      if (ret == 0) {
        std::chrono::duration<double, std::micro> micro = end - start;
        times.push_back(micro.count());
        // profile_path.push_back(tmp_dir);
      }
    }
    // std::vector<double> times =
    //     run_autotune_script(py_script_, profile_path, chip_code, mode_);
    int n = 0;
    int best_idx = -1;
    double best_time = std::numeric_limits<double>::max();
    if (mode_ == "cmodel") {
      std::cout
          << "\033[31m[PPL_TUNE] Your are tuned under cmodel the results "
             "may not be reliable, please re-run with '--mode pcie'\033[0m\n";
    }
    for (int i = 0; i < fc.param.configs.size(); i++) {
      // print tune result
      std::string log = "";
      for (auto param : fc.param.tunable)
        log += " " + param + ": " +
               variant_to_string(fc.param.configs[i].at(param));
      std::cout << "\033[32m[PPL_TUNE] " + func_name << " ret = " << fc.rets[i]
                << " with tunable params" << log << " time: "
                << (fc.rets[i] ? "NULL" : std::to_string(times[n] / iter_num_))
                << "(us)\033[0m\n";
      // get the best
      if (fc.rets[i])
        continue;
      fc.times[i] = times[n++];
      if (fc.times[i] != 0 && fc.times[i] < best_time) {
        best_time = fc.times[i];
        best_idx = i;
      }
    }
    // clean up workdir
    fs::remove_all(path);
    // store config
    if (best_idx != -1) {
      auto &config = fc.param.configs[best_idx];
      std::string content = "";
      std::string fixed_param = "";
      std::string tunable_param = "";
      for (auto param : fc.param.fixed) {
        if (!fixed_param.empty())
          fixed_param += ",";
        fixed_param += param;
      }
      for (auto param : fc.param.tunable) {
        if (!tunable_param.empty())
          tunable_param += ",";
        tunable_param += param;
        if (!content.empty())
          content += ",";
        content += variant_to_string(config.at(param));
      }
      if (!content.empty()) {
        std::ofstream config_file(path.concat(".config"));
        if (!config_file.is_open()) {
          throw std::runtime_error("Creat " + path.string() + "file failed.");
          return -1;
        }
        config_file << fixed_param << std::endl;
        config_file << tunable_param << std::endl;
        config_file << content;
        config_file.close();
        std::cout << "Autotune result stored in : " << path << std::endl;
        if (!std::getenv("AUTOTUNE_CACHE_PATH"))
          std::cout << "\033[33mUse default path: " << cache_path_
                    << "\nYou can use env 'AUTOTUNE_CACHE_PATH=path' to "
                       "specify the path where "
                       "the result is stored.\033[0m"
                    << std::endl;
      }
    }
    if (times.size() == 0)
      return ret;
    return 0;
  }
#endif
  template <typename... Args>
  void get(const std::string &func_name, const std::string &ir_md5,
           std::string chip_code, const std::vector<std::string> &fixed,
           const std::vector<std::string> &tunable, Args... args) {
    if (functions_.count(func_name)) {
      return;
    }
    // get real chip_code for ppl_jit
    if (getenv("CHIP"))
      chip_code = get_chip_code();
    std::vector<std::any> default_args = {args...};
    std::string key =
        chip_code + "_" + func_name + "_" + ir_md5.substr(MD5_LENS);
    for (int i = 0; i < fixed.size(); i++) {
      key += "-" + any_to_string(default_args[i]);
    }
    if (!global_cache_.count(key))
      return;
    auto param = global_cache_[key];
    if (param.fixed != fixed || param.tunable != tunable)
      return;
    int offset = fixed.size();
    for (int i = 0; i < tunable.size(); i++)
      update_arg(default_args, offset + i, param.configs[0][tunable[i]]);
  }

  void load_cache() {
    if (!fs::exists(cache_path_)) {
      return;
    }
    for (const auto &chip_dir_entry : fs::directory_iterator(cache_path_)) {
      if (!chip_dir_entry.is_directory())
        continue;
      std::string chip_code = chip_dir_entry.path().filename().string();
      for (const auto &entry : fs::directory_iterator(chip_dir_entry.path())) {
        if (!entry.is_regular_file() || entry.path().extension() != ".config")
          continue;
        auto path = entry.path();
        auto filename = path.stem().string();
        std::string key = chip_code + "_" + filename;
        std::ifstream ifs(path);
        if (!ifs)
          continue;
        int line_num = 0;
        std::string line;
        TuneParam _param;
        while (std::getline(ifs, line)) {
          std::istringstream ss(line);
          std::string token;
          TuneConfig _config;
          for (int i = 0; std::getline(ss, token, ','); ++i) {
            if (line_num == 0)
              _param.fixed.push_back(token);
            else if (line_num == 1)
              _param.tunable.push_back(token);
            else if (line_num == 2) {
              _config[_param.tunable[i]] = string_to_variant(token.c_str());
            }
          }
          if (line_num >= 2) {
            _param.configs.push_back(_config);
          }
          ++line_num;
        }
        if (global_cache_.count(key) == 0) {
          std::cout << "Load autotune cache " << chip_dir_entry.path() / key << "\n";
        }
        global_cache_[key] = _param;
      }
    }
  }

private:
#ifdef PPL_TUNE
  int call_func(const std::function<int(const std::vector<std::any> &)> &func,
                const std::vector<std::any> &current_args,
                const std::string &chip_code, const std::string &sub,
                int iter_num, bool profile = true) {
    tpudnnHandle_t t_handle = std::any_cast<tpudnnHandle_t>(current_args[0]);
    if (profile) {
      if (mode_ == "cmodel")
        setenv("FILE_DUMP_CMD", sub.c_str(), 1);
      else {
        if (get_runtime_type(chip_code) == RUNTIME_TPUV7)
          tpudnnEnableProfile(t_handle, record_size_, 1);
      }
    }
    int ret;
    for (int i = 0; i < iter_num; ++i) {
      ret = func(current_args);
      if (ret != 0)
        break;
    }
    tpudnnSync(t_handle);
    if (profile) {
      if (mode_ == "cmodel")
        unsetenv("FILE_DUMP_CMD");
      else {
        if (get_runtime_type(chip_code) == RUNTIME_TPUV7)
          tpudnnDisableProfile(t_handle);
      }
    }
    return ret;
  }

  // from vector<any> to tuple
  template <size_t I = 0, typename Tuple>
  static void fill_tuple_from_vector(Tuple &tuple,
                                     const std::vector<std::any> &vec) {
    if constexpr (I < std::tuple_size_v<Tuple>) {
      using T = std::remove_reference_t<std::tuple_element_t<I, Tuple>>;
      std::get<I>(tuple) = std::any_cast<T>(vec[I]);
      fill_tuple_from_vector<I + 1>(tuple, vec);
    }
  }

  std::string serialize_fixed(const std::vector<std::any> &args,
                              const TuneParam &param) {
    std::string s = "";
    for (size_t i = 0; i < param.fixed.size(); ++i) {
      auto it = std::find(param.full_param.begin(), param.full_param.end(),
                          param.fixed[i]);
      if (it != param.full_param.end()) {
        size_t pos = it - param.full_param.begin();
        s += "-" + any_to_string(args[pos]);
      }
    }
    return s;
  }

  // convert variant to string
  std::string variant_to_string(const ParamValue &value) {
    if (std::holds_alternative<int>(value)) {
      return std::to_string(std::get<int>(value));
    } else if (std::holds_alternative<unsigned long long>(value)) {
      return std::to_string(std::get<unsigned long long>(value));
    } else if (std::holds_alternative<float>(value)) {
      return std::to_string(std::get<float>(value));
    } else if (std::holds_alternative<double>(value)) {
      return std::to_string(std::get<double>(value));
    } else if (std::holds_alternative<bool>(value)) {
      return std::get<bool>(value) ? "true" : "false";
    } else if (std::holds_alternative<std::string>(value)) {
      return std::get<std::string>(value);
    }
    return "unknown";
  }

  std::any variant_to_any(const ParamValue &value) {
    if (std::holds_alternative<int>(value)) {
      return std::any(std::get<int>(value));
    } else if (std::holds_alternative<unsigned long long>(value)) {
      return std::any(std::get<unsigned long long>(value));
    } else if (std::holds_alternative<float>(value)) {
      return std::any(std::get<float>(value));
    } else if (std::holds_alternative<double>(value)) {
      return std::any(std::get<double>(value));
    } else if (std::holds_alternative<bool>(value)) {
      return std::any(std::get<bool>(value));
    } else if (std::holds_alternative<std::string>(value)) {
      return std::any(std::get<std::string>(value));
    } else if (std::holds_alternative<tpudnnHandle_t>(value)) {
      return std::any(std::get<tpudnnHandle_t>(value));
    }
    throw std::runtime_error("Unsupported type in variant");
  }
#endif

  void update_arg(std::vector<std::any> &default_args, size_t pos,
                  const ParamValue &v) {
    if (default_args[pos].type() == typeid(int *)) {
      *std::any_cast<int *>(default_args[pos]) = std::get<int>(v);
    } else if (default_args[pos].type() == typeid(unsigned long long *)) {
      *std::any_cast<unsigned long long *>(default_args[pos]) =
          std::get<unsigned long long>(v);
    } else if (default_args[pos].type() == typeid(float *)) {
      *std::any_cast<float *>(default_args[pos]) = std::get<float>(v);
    } else if (default_args[pos].type() == typeid(double *)) {
      *std::any_cast<double *>(default_args[pos]) = std::get<double>(v);
    } else if (default_args[pos].type() == typeid(bool *)) {
      *std::any_cast<bool *>(default_args[pos]) = std::get<bool>(v);
    } else {
      throw std::runtime_error("Unsupported type in variant");
    }
  }

  // convert any to string
  std::string any_to_string(const std::any &a) {
    if (a.type() == typeid(int))
      return std::to_string(std::any_cast<int>(a));
    if (a.type() == typeid(unsigned long long))
      return std::to_string(std::any_cast<unsigned long long>(a));
    if (a.type() == typeid(float))
      return std::to_string(std::any_cast<float>(a));
    if (a.type() == typeid(double))
      return std::to_string(std::any_cast<double>(a));
    if (a.type() == typeid(std::string))
      return std::any_cast<std::string>(a);
    if (a.type() == typeid(bool))
      return std::any_cast<bool>(a) ? "true" : "false";
    // if (a.type() == typeid(tpudnnHandle_t)) {
    //   std::stringstream ss;
    //   ss << "0x" << std::hex
    //      << reinterpret_cast<uintptr_t>(std::any_cast<tpudnnHandle_t>(a));
    //   return ss.str();
    // }
    return "unknown";
  }

  ParamValue string_to_variant(const std::string &str) {
    // Try int
    try {
      size_t pos;
      int val = std::stoi(str, &pos);
      if (pos == str.size())
        return val;
    } catch (...) {
    }
    // Try unsigned long long
    try {
      size_t pos;
      unsigned long long val = std::stoull(str, &pos);
      if (pos == str.size())
        return val;
    } catch (...) {
    }
    // Try float
    try {
      size_t pos;
      float val = std::stof(str, &pos);
      if (pos == str.size())
        return val;
    } catch (...) {
    }
    // Try double
    try {
      size_t pos;
      double val = std::stod(str, &pos);
      if (pos == str.size())
        return val;
    } catch (...) {
    }
    // Try bool
    if (str == "true")
      return true;
    if (str == "false")
      return false;
    throw std::runtime_error("Unsupported convertion" + str + "to variant");
    return str;
  }

  RuntimeType get_runtime_type(const std::string &chip_code) {
    static const std::set<std::string> rtv7_chips = {"tpub_7_1", "tpub_7_1_e",
                                                     "tpub_9_0"};
    static const std::set<std::string> bmlib_chips = {
        "tpu_6_0", "tpu_6_0_e", "tpul_6_0", "tpub_9_3", "tpul_8_1"};
    if (rtv7_chips.count(chip_code))
      return RUNTIME_TPUV7;
    if (bmlib_chips.count(chip_code))
      return RUNTIME_BMLIB;
    throw std::runtime_error("unspport chip_code: " + std::string(chip_code));
    return RUNTIME_UNKNOW;
  }

  std::map<std::string, TuneParam> global_cache_;
  std::map<std::string, FunctionConfig> functions_;
  fs::path cache_path_;
  fs::path py_script_;
  std::string mode_;
  int record_size_;
  int iter_num_;
};
// define global tunner
extern AutoTuner gTunner;
#endif

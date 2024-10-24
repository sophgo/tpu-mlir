#ifndef __PPL_GIT_H__
#define __PPL_GIT_H__
#include <algorithm>
#include <cassert>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

enum PplErrorCode {
  PplAddressAssignErr = 0x1001,
  FileErr = 0x1002,
  LlvmFeErr = 0x1003,
  PplFeErr = 0x1004,
  PplOpt1Err = 0x1005,
  PplOpt2Err = 0x1006,
  PplFinalErr = 0x1007,
  PplTransErr = 0x1008,
  EnvErr = 0x1009,
};

namespace fs = std::filesystem;

typedef void (*KERNEL_FUNC)(void *);
typedef void (*NODE_FUNC)(void *);
#define MaxCacheRetained 100
using Value =
    std::tuple<std::string, void *, std::unordered_map<std::string, NODE_FUNC>>;
class LRUCache {
public:
  using Key = std::size_t;
  static LRUCache &getInstance(const std::string &dir,
                               std::size_t cache_items) {
    static LRUCache instance(dir, cache_items);
    return instance;
  }

  LRUCache(const LRUCache &) = delete;
  LRUCache &operator=(const LRUCache &) = delete;
  bool empty() { return LRU.empty(); }
  void put(const Key &K, Value V) {
    auto clean_dir = [](Value &v) {
      auto &[path, handle, _] = v;
      if (handle)
        dlclose(handle);
      std::stringstream cmd;
      cmd << "rm -rf " << path;
      system(cmd.str().c_str());
    };
    std::unique_lock<std::mutex> Lock(mtx);
    assert(findByKey(K) == LRU.end());

    LRU.insert(LRU.begin(), {K, V});
    if (LRU.size() <= MaxSize)
      return;
    clean_dir(LRU.back().second);
    LRU.pop_back();
  }

  std::optional<Value> take(Key K) {
    std::unique_lock<std::mutex> Lock(mtx);
    auto Existing = findByKey(K);
    if (Existing == LRU.end()) {
      return std::nullopt;
    }

    std::rotate(LRU.begin(), Existing, Existing + 1);
    return std::optional<Value>(LRU.begin()->second);
  }

  void update(const Key &K, Value V) {
    std::unique_lock<std::mutex> Lock(mtx);
    auto Existing = findByKey(K);
    (*Existing).second = V;
    return;
  }

private:
  using KVPair = std::pair<Key, Value>;
  explicit LRUCache(std::string dir, std::size_t MaxSize)
      : dir(std::move(dir)), MaxSize(MaxSize) {}
  std::vector<KVPair>::iterator findByKey(Key K) {
    return std::find_if(LRU.begin(), LRU.end(),
                        [K](const KVPair &P) { return P.first == K; });
  }

  std::mutex mtx;
  std::string dir;
  std::size_t MaxSize;
  std::vector<KVPair> LRU;
};

static int ppl_jit_call(const char *file_name, const char *func_name,
                        const char *args, void *st, const char *chip,
                        void *pid_node) {
  static std::map<std::size_t, std::size_t> state;
  auto convertToKey = [](const std::string &str) {
    try {
      std::size_t pos;
      unsigned long long converted = std::stoull(str, &pos);
      if (pos < str.length()) {
        return false;
      }

      return converted <= std::numeric_limits<std::size_t>::max();
    } catch (const std::invalid_argument &e) {
      std::cerr << "Invalid argument: " << e.what() << std::endl;
    } catch (const std::out_of_range &e) {
      std::cerr << "Out of range: " << e.what() << std::endl;
    }
    return false;
  };

  auto gen_flag = [&](const std::string &cache_path, std::size_t key,
                      std::size_t flag) {
    std::string filePath = cache_path + "flag.txt";
    std::ofstream file(filePath, std::ios::app);
    file << key << " " << flag << std::endl;
    state[key] = flag;
    file.close();
  };

  auto init_flag = [&](const std::string &cache_path) {
    std::string filePath = cache_path + "flag.txt";
    std::ifstream file(filePath);
    std::string line;
    std::size_t key, flag;

    while (std::getline(file, line)) {
      std::istringstream iss(line);
      if (!(iss >> key >> flag)) {
        continue;
      }
      state[key] = flag;
    }

    file.close();
    return;
  };

  std::string work_dir;
  if (auto env = getenv("PPL_WORK_PATH")) {
    work_dir = std::string(env);
  } else {
    work_dir = std::string(getenv("PWD"));
  }
  std::string ppl_root = getenv("PPL_PROJECT_ROOT");
  std::string cache_dir;
  if (auto env = getenv("PPL_CACHE_PATH"); env)
    cache_dir = std::string(env);

  if (cache_dir.empty()) {
    cache_dir = getenv("HOME") + std::string("/.ppl/cache/");
  } else {
    std::filesystem::path p(cache_dir);
    if (p.has_filename() && p.filename().string() != "/") {
      cache_dir += std::string("/");
    }
  }

  if (!fs::exists(cache_dir))
    fs::create_directories(cache_dir);
  std::size_t cache_items = MaxCacheRetained;

  if (auto item = getenv("CACHE_ITEMS"); item) {
    if (convertToKey(std::string(item))) {
      cache_items = static_cast<std::size_t>(std::stoull(std::string(item)));
    }
  }
  LRUCache &cache = LRUCache::getInstance(cache_dir, cache_items);
  // construct the cached item by search the directory
  if (cache.empty()) {
    for (const auto &entry : fs::directory_iterator(cache_dir)) {
      if (fs::is_directory(entry.status())) {
        auto filename = entry.path().filename();
        if (convertToKey(filename.string())) {
          auto val =
              std::make_tuple(entry.path().string() + std::string("/"), nullptr,
                              std::unordered_map<std::string, NODE_FUNC>());
          cache.put(static_cast<std::size_t>(std::stoull(filename.string())),
                    val);
        }
      }
    }
  }

  init_flag(cache_dir + "/");
  std::string str = std::string(chip) + std::string(file_name) +
                    std::string(func_name) + std::string(args);
  std::size_t key = std::hash<std::string>{}(str);
  auto v = cache.take(key);
  if (!v && !state.count(key)) {
    std::string inc_path = ppl_root + "/inc";
    std::stringstream cmd;
    auto path = cache_dir + std::to_string(key) + "/";
    v = {path, nullptr, {}};
    std::filesystem::create_directory(path);
    cmd << "ppl_jit.sh " << std::quoted(file_name) << " " << func_name << " "
        << inc_path << " " << path << " " << chip << " " << args
        << " > nul 2>nul";
    auto ret = system(cmd.str().c_str());
    switch (ret) {
    case 0: {
      gen_flag(cache_dir + "/", key, true);
      break;
    }
    case PplAddressAssignErr: {
      fs::remove_all(path);
      gen_flag(cache_dir + "/", key, false);
      return ret;
    }
    default: {
      cmd << "ppl_jit.sh " << std::quoted(file_name) << " " << func_name << " "
          << inc_path << " " << path << " " << chip << " " << args;
      fs::remove_all(path);
      gen_flag(cache_dir + "/", key, false);
      return system(cmd.str().c_str());
    }
    }
    cache.put(key, *v);
  }

  if (!state[key])
    return -1;

  if (!std::get<1>(*v)) {
    auto kernel_so_name = std::get<0>(*v) + std::string("lib/lib") +
                          std::string(func_name) + std::string(".so");
    auto handle = dlopen(kernel_so_name.c_str(), RTLD_NOW);
    if (!handle) {
      printf("open ppl kernel so failed! %s\n", dlerror());
      return -2;
    }
    auto set_id = (NODE_FUNC)dlsym(handle, "set_id_node");
    set_id(pid_node);
    auto kernel_func = (KERNEL_FUNC)dlsym(handle, func_name);
    if (!kernel_func) {
      printf("get ppl kernel func failed! %s\n", dlerror());
      dlclose(handle);
      return -2;
    }
    kernel_func(st);
    auto get_id = (NODE_FUNC)dlsym(handle, "get_id_node");
    get_id(pid_node);
    auto [path, ignored1, _] = (*v);
    std::unordered_map<std::string, NODE_FUNC> symbolMap = {
        {std::string(func_name), kernel_func},
        {"set_id_node", set_id},
        {"get_id_node", get_id}};
    auto val = std::make_tuple(path, handle, std::move(symbolMap));
    cache.update(key, val);
  } else {
    auto iter = std::get<2>(*v).find("set_id_node");
    iter->second(pid_node);
    iter = std::get<2>(*v).find(std::string(func_name));
    iter->second(st);
    iter = std::get<2>(*v).find("get_id_node");
    iter->second(pid_node);
  }

  return 0;
}

#endif

#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include <ppl_jit.h>
#include "host_def.h"


#ifdef __cplusplus
extern "C" {
#endif
static thread_local void *m_cmdid_node = NULL;

void ppl_set_node(void *cmdid_node)
{
    m_cmdid_node = cmdid_node;
}

static int execCompileCommand(const std::string &command, std::string &output) {
  char buffer[256];
  FILE *pipe = popen(command.c_str(), "r");
  if (!pipe)
    return -1;

  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }

  return pclose(pipe);
}

namespace fs = std::filesystem;
const std::string RESULT_FILE = "compiled";

int acquireLock(const fs::path& lockPath) {
    int fd = open(lockPath.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        throw std::runtime_error("Procecss " + std::to_string(getpid()) +
                                 ": Failed to open: " + lockPath.string());
    }
    if (flock(fd, LOCK_EX) == -1) {
        close(fd);
        throw std::runtime_error("Procecss " + std::to_string(getpid()) +
                                 ": File lock failure: " + lockPath.string());
    }
    return fd;
}

void releaseLock(int fd) {
    flock(fd, LOCK_UN);
    close(fd);
}

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
    auto v = LRU.begin()->second;
    if (!std::get<1>(v)) {
      if (!fs::exists(fs::path(std::get<0>(v)) / RESULT_FILE)) {
        LRU.erase(LRU.begin());
        return std::nullopt;
      }
    }
    return std::optional<Value>(v);
  }

  void update(const Key &K, Value V) {
    std::unique_lock<std::mutex> Lock(mtx);
    auto Existing = findByKey(K);
    (*Existing).second = V;
    return;
  }

protected:
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

class PplJitCache : public LRUCache {
public:
  PplJitCache(const std::string &cache_dir, std::size_t max_items)
      : LRUCache(cache_dir, max_items), cache_dir(cache_dir) {
    initFlags();
    initializeCache();
  }

  PplJitCache(const PplJitCache &) = delete;
  PplJitCache &operator=(const PplJitCache &) = delete;

  int jitCompile(const std::string &file_name, const std::string &func_name,
                 const std::string &args, const std::string &chip,
                 std::size_t key) {
    auto v = take(key);
    if (!v && !state.count(key)) {
      int ret = runJitCompile(file_name, func_name, args, chip, key);
      if (ret == 0) {
        generateFlag(key, ret);
        auto val =
            std::make_tuple(cache_dir + std::to_string(key) + "/", nullptr,
                            std::unordered_map<std::string, NODE_FUNC>());
        put(key, val);
#ifdef DDEBUG
        printf("[compile jit success]\n");
#endif
      } else
        generateFlag(key, ret);
      return ret;
    }
    return 0;
  }

  int isCompiled(std::size_t key) const { return state.at(key); }

  int loadAndExecute(std::optional<Value> &v, std::size_t key,
                     const std::string &func_name, void *st, void *pid_node) {
    if (!std::get<1>(*v)) {
      auto kernel_so_name =
          std::get<0>(*v) + std::string("lib/lib") + func_name + ".so";
      auto handle = dlopen(kernel_so_name.c_str(), RTLD_NOW);
      if (!handle) {
        printf("open ppl kernel so failed! %s\n", dlerror());
        return CacheOpenKernelSoErr;
      }
      auto set_id = (NODE_FUNC)dlsym(handle, "set_id_node");
      set_id(pid_node);
      auto kernel_func = (KERNEL_FUNC)dlsym(handle, func_name.c_str());
      if (!kernel_func) {
        printf("get ppl kernel func failed! %s\n", dlerror());
        dlclose(handle);
        return CacheGetKernelFunErr;
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
      update(key, val);
    } else {
      auto iter = std::get<2>(*v).find("set_id_node");
      iter->second(pid_node);
      iter = std::get<2>(*v).find(func_name);
      iter->second(st);
      iter = std::get<2>(*v).find("get_id_node");
      iter->second(pid_node);
    }
#ifdef DDEBUG
    printf("[run success]\n");
#endif
    return 0;
  }

private:
  std::string cache_dir;
  std::map<std::size_t, int> state;

  void initFlags() {
    std::ifstream file(cache_dir + "flag.txt");
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
  }

  void generateFlag(std::size_t key, std::size_t flag) {
    std::ofstream file(cache_dir + "flag.txt", std::ios::app);
    std::string data = std::to_string(key) + " " + std::to_string(flag) + "\n";
    file << data;
    state[key] = flag;
  }

  void initializeCache() {
    for (const auto &entry : fs::directory_iterator(cache_dir)) {
      if (fs::is_directory(entry.status())) {
        auto filename = entry.path().filename();
        try {
          std::size_t key = std::stoull(filename.string());
          auto val =
              std::make_tuple(entry.path().string() + "/", nullptr,
                              std::unordered_map<std::string, NODE_FUNC>());
          put(key, val);
        } catch (...) {
          continue;
        }
      }
    }
  }

  int runJitCompile(const std::string &file_name, const std::string &func_name,
                    const std::string &args, const std::string &chip,
                    std::size_t key) {
    std::string inc_path = std::string(getenv("PPL_PROJECT_ROOT")) + "/inc";
    std::string path = cache_dir + std::to_string(key) + "/";
    std::string lock_file = "/tmp/ppl_" + std::to_string(key) + ".lock";
    fs::create_directory(path);
    fs::path result_file = fs::path(path) / RESULT_FILE;
    int ret = 0;
    if (fs::exists(result_file)) {
        return ret;
    }
    // file lock
    int lockFd = -1;
    lockFd = acquireLock(lock_file);
    if (!fs::exists(result_file)) {
      // compile
      std::stringstream cmd;
      cmd << "ppl_jit.sh " << std::quoted(file_name) << " " << func_name << " "
          << inc_path << " " << path << " " << chip << " " << args
          << " > /dev/null 2>&1\n";
      std::string output;
      ret = system(cmd.str().c_str());
      ret >>= 8;
      switch (ret) {
      case 0: {
      #ifdef DDEBUG
        printf("[compile jit success]\n");
      #endif
        std::ofstream ofs(result_file);
        if (!ofs) {
          throw std::runtime_error(
            "Procecss " + std::to_string(getpid()) +
            ": Creat flag file failed: " + result_file.string());
          ofs.close();
        }
        break;
      }
      case PplL2AddrAssignErr:
      case PplLocalAddrAssignErr: {
        fs::remove_all(path);
        break;
      }
      default: {
        cmd.str("");
        cmd << "ppl_jit.sh " << std::quoted(file_name) << " " << func_name << " "
            << inc_path << " " << path << " " << chip << " " << args;
        system(cmd.str().c_str());
        fs::remove_all(path);
        break;
      }
      }
    }
    releaseLock(lockFd);
    return ret;
  }
};

int ppl_jit_call(const char *file_name, const char *func_name,
                  const char *args, void *st) {
  std::string work_dir =
      getenv("PPL_WORK_PATH") ? getenv("PPL_WORK_PATH") : getenv("PWD");
  std::string cache_dir = getenv("PPL_CACHE_PATH")
                              ? getenv("PPL_CACHE_PATH")
                              : std::string(getenv("HOME")) + "/.ppl/cache/";
  // Todo. check mchip m_cmdid_node if torch don't need this param
  // if (!m_cmdid_node || !m_chip) {
  auto chip = get_chip_str();
  if (!chip) {
    return PplJitSetChipErr;
  }
  if (!cache_dir.empty() && cache_dir.back() != '/') {
    cache_dir += "/";
  }
  if (!fs::exists(cache_dir)) {
    fs::create_directories(cache_dir);
  }

  std::size_t cache_items =
      getenv("CACHE_ITEMS")
          ? static_cast<std::size_t>(std::stoull(getenv("CACHE_ITEMS")))
          : MaxCacheRetained;

  static PplJitCache cache(cache_dir, cache_items);

  std::string str = std::string(chip) + file_name + func_name + args;
  std::size_t key = std::hash<std::string>{}(str);

  auto ret = cache.jitCompile(file_name, func_name, args, chip, key);
  if (ret != 0)
    return ret;
  auto v = cache.take(key); // state
  if (!v) {
    return cache.isCompiled(key);
  }

  return cache.loadAndExecute(v, key, func_name, st, m_cmdid_node);
}
#ifdef __cplusplus
}
#endif

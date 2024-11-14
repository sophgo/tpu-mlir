#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/GenericCpuFunc.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "utils.hpp"
#include <assert.h>
#include <iostream>
#include <map>
#include <memory>
using namespace tpu_mlir;
namespace cvi_debug {
class FieldBase {
public:
  FieldBase(TypeID typeID) : typeID(typeID){};
  virtual ~FieldBase() = default;
  TypeID typeID;
};

template <typename T>
class Field : public FieldBase {
public:
  Field(T &val) : data(val), FieldBase(TypeID::get<Field<T>>()) {}
  static bool classof(const FieldBase *base) {
    return base->typeID == TypeID::get<Field<T>>();
  }
  T data;
};

class OpParam {
public:
  template <typename T>
  void put(std::string name, T value) {
    fields[name] = std::make_shared<Field<T>>(value);
  }

  template <typename T>
  T &get(std::string name) {
    // llvm::errs()<<"name:"<<name<<",value:"<<fields[name].get()<<"\n";
    auto f = dyn_cast<Field<T>>(fields[name].get());
    assert(f);
    return f->data;
  }

  bool has(std::string name) {
    auto it = fields.find(name);
    return it != fields.end();
  }

private:
  std::map<std::string, std::shared_ptr<FieldBase>> fields;
};

struct io_mem_info {
  std::string name;
  std::string type;
  double qscale;
  uint64_t gaddr;
  std::vector<int64_t> g_shape;
  uint32_t count; // num of elements
  uint32_t size;  // count * typesize
};

struct cpu_func_info {
  std::string func_name;
  OpParam params;
  std::vector<io_mem_info> inputs;
  std::vector<io_mem_info> outputs;
};

void handleFuncArgs(const uint8_t *args, OpParam &param);

void getCpuInput(std::vector<std::vector<float>> &inputs, uint8_t *vaddr,
                 int64_t &offset, io_mem_info &info);

void getAndSaveCpuOutput(std::vector<float> &output,
                         std::vector<float> &save_data, uint8_t *vaddr,
                         io_mem_info &info);

void invoke(cpu_func_info &func_info, std::vector<std::vector<float>> &inputs,
            std::vector<std::vector<float>> &outputs);
} // namespace cvi_debug

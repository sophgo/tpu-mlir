#pragma once

#include "llvm/Support/DynamicLibrary.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

struct cmd_id_node;
typedef struct cmd_id_node CMD_ID_NODE;

typedef enum {
  STORAGE_MODE_1N_FP32    = 0,
  STORAGE_MODE_1N_INT8    = 1,
  STORAGE_MODE_1N_INT16   = 2,
  STORAGE_MODE_2N_INT16   = 3,
  STORAGE_MODE_4N_INT8    = 4,
  STORAGE_MODE_2IC_FP32   = 5,  // special for 2IC weight
  STORAGE_MODE_4N_4IC_4OC = 6,
  STORAGE_MODE_4N_INT16   = 7,
  STORAGE_MODE_UNINITILIZED,
  STORAGE_MODE_END
} TENSOR_STORAGE_MODE;

typedef enum {
  ROUND_INF = 0,     // 1.5 -> 2   -1.5 -> -2
  ROUND_UP = 1,      // 1.5 -> 2   -1.5 -> -1
  ROUND_DOWN = 2,    // 1.5 -> 1   -1.5 -> -2
  ROUND_EVEN = 3,    // 1.5 -> 2    2.5 -> 2
  ROUND_ODD = 4,     // 1.5 -> 1    0.5 -> 1
  ROUND_ZERO = 5,    // 1.5 -> 1   -1.5 -> -1
  TRIM_ZERO = 6,     // 1.6 -> 1   -1.6 -> -1
  TRIM_INF = 7,      // 1.4 -> 2   -1.4 -> -2
  TRIM_UP = 8,       // 1.4 -> 2   -1.6 -> -1
  TRIM_DOWN = 9,     // 1.6 -> 1   -1.4 -> -2
} ROUND_MODE;

typedef enum {
  STORE_MODE_1N = 0,
  STORE_MODE_2N = 1,
  STORE_MODE_4N = 2,
} STORE_MODE_T;

#define BM_BINARY_ADD 0
#define BM_BINARY_SUB 1
#define BM_BINARY_MUL 2
#define BM_BINARY_DIV 3
#define BM_BINARY_MAX 4

#define SUBNET_MODE_TPU 0
#define SUBNET_MODE_CPU 1
#define SUBNET_MODE_MERGE 2
#define SUBNET_MODE_SWITCH 3

#define MEM_TYPE_TPU (1 << 0)
#define MEM_TYPE_CPU (1 << 1)
#define MEM_TYPE_ALL (MEM_TYPE_TPU | MEM_TYPE_CPU)

typedef enum {
  DTYPE_FP32 = 0,
  DTYPE_FP16 = 1,
  DTYPE_INT8 = 2,
  DTYPE_UINT8 = 3,
  DTYPE_INT16 = 4,
  DTYPE_UINT16 = 5,
  DTYPE_INT32 = 6,
  DTYPE_UINT32 = 7,
  DTYPE_BFP16 = 8,
  DTYPE_UNKNOWN = -1,
} DATA_TYPE_T;
typedef DATA_TYPE_T bm_data_type_t;

typedef struct bmcompiler_mem_info {
    uint64_t addr;
    uint64_t size;
    uint64_t offset;
} bm_mem_desc_t;
typedef struct bmcompiler_mem_info bm_device_mem_t;

typedef int (*cmodel_init)(int node_idx, unsigned long long global_mem_size);
typedef void (*cmodel_deinit)(int node_idx);
typedef void *(*get_global_memaddr)(int node_idx);
typedef void (*set_cmd_buffer_ptr)(void *gdma_buffer_ptr, void *bdc_buffer_ptr);

namespace sophgo {
namespace backend {
class BM168x {

public:
  virtual ~BM168x() = 0;
  cmodel_init dl_cmodel_init;
  cmodel_deinit dl_cmodel_deinit;
  get_global_memaddr dl_get_global_memaddr;
  set_cmd_buffer_ptr dl_set_cmd_buffer_ptr;

  CMD_ID_NODE *get_cmd_id_node() { return (CMD_ID_NODE *)cmdid_node; }
  static bm_data_type_t getType(mlir::Type type);
  void *get_gmem_addr(uint64_t addr);
  void *get_gmem_addr(const bm_device_mem_t &mem);
  void bm_memcpy_s2d(const bm_device_mem_t &dst, void *src);
  void bm_memcpy_d2s(void *dst, const bm_device_mem_t &src);
  void value_s2d(mlir::Value v, void *src);
  void value_d2s(mlir::Value v, void *dst);

  // arch info
  virtual uint64_t get_gmem_start() = 0;
  virtual uint64_t get_ctx_start_addr() = 0;
  uint64_t get_cmodel_gmem_size();

  static bm_data_type_t getDataType(mlir::Value v);

public:
  std::shared_ptr<std::vector<uint32_t>> bdc_buffer;
  std::shared_ptr<std::vector<uint32_t>> gdma_buffer;
  uint32_t gdma_total_id;
  uint32_t bdc_total_id;
  std::vector<uint32_t> gdma_group_id;
  std::vector<uint32_t> bdc_group_id;
  int cmdid_groupnum;

  static const int64_t ALIGNMENT = 0x1000;

protected:
  void *cmdid_node;
  void *bdc_node;
  void *gdma_node;
  bool really_issue_command;
  llvm::sys::DynamicLibrary DL;
};

}
}

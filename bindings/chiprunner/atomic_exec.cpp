//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "bmlib_runtime.h"
#include <iostream>
#include <string>

#ifndef WIN32
#define WITH_PLATFORM(x) __attribute__((packed)) x
#else
#define WITH_PLATFORM(x) x
#endif

typedef struct {
  int length;
  int command_type;
  char data[128];
} WITH_PLATFORM(sg_api_atomic_t);

#define MAX_CMD_NUM 4

typedef struct {
  int cmd_num;
  int command_type[MAX_CMD_NUM];
  int command_length[MAX_CMD_NUM];
  char data[128 * MAX_CMD_NUM];
} WITH_PLATFORM(sg_api_atomics_t);

typedef unsigned long long u64;
typedef unsigned int u32;

#define ASSERT_SUCCESS(command)                                                \
  {                                                                            \
    bm_status_t status_;                                                       \
    status_ = command;                                                         \
    if (status_ != BM_SUCCESS) {                                               \
      std::cerr << #command << "  " << __func__ << ":" << __LINE__             \
                << "   command failed" << std::endl;                           \
      exit(-1);                                                                \
    }                                                                          \
  }

class ChipRunner {
public:
  ChipRunner(const char *fn, int devid = 0) {
    bm_dev_request(&bm_handle, devid);
    tpu_module = tpu_kernel_load_module_file(bm_handle, fn);
    init_reserved_memory();
  }
  ChipRunner(const char *data, size_t length, int devid = 0) {
    bm_dev_request(&bm_handle, devid);
    tpu_kernel_load_module(bm_handle, data, length);
    init_reserved_memory();
  }

  int init_reserved_memory() {
    auto ret =
        bm_malloc_device_byte(bm_handle, &device_mem, 1024 * 1024 * 1024);
    u64 res_addr = bm_mem_get_device_addr(device_mem);
    set_reserved_mem(res_addr);
    return (int)ret;
  }

  ~ChipRunner() {
    if (tpu_module) {
      tpu_kernel_unload_module(bm_handle, (tpu_kernel_module_t)tpu_module);
      tpu_module = 0;
    }
    if (bm_handle) {
      bm_free_device(bm_handle, device_mem);
      bm_dev_free(bm_handle);
    }
  }

  void set_reserved_mem(u64 _reserved_offset) {
    reserved_offset = _reserved_offset;
  }
  u64 get_reserved_mem() { return reserved_offset; }

  int chip_s2d(u64 address, size_t size, void *data) {
    bm_device_mem_t device_mem = bm_mem_from_device(address, size);

    auto ret = bm_memcpy_s2d(bm_handle, device_mem, (void *)data);
    // std::cout << "save to device " << address << std::endl;
    return (int)ret;
  }

  int chip_d2s(u64 address, size_t size, void *data) {
    bm_device_mem_t device_mem = bm_mem_from_device(address, size);
    // std::cout << "copy from device " << address << std::endl;
    auto ret = bm_memcpy_d2s(bm_handle, (void *)data, device_mem);
    return (int)ret;
  }

  int launch_single_cmd(void *command, int command_type, size_t size) {
    tpu_kernel_function_t func_id;

    sg_api_atomic_t params = {0};
    params.command_type = command_type;
    for (size_t i = 0; i < size; i++) {
      params.data[i] = ((char *)command)[i];
    }
    params.length = size;
    func_id = tpu_kernel_get_function(bm_handle, tpu_module,
                                      "tpu_kernel_launch_atomic");

    // std::cout << func_id << std::endl;
    auto ret = tpu_kernel_launch(bm_handle, func_id, &params, sizeof(params));

    return (int)ret;
  }

  int launch_cmds(void *command, void *_command_type, void *command_length,
                  int cmd_num) {
    sg_api_atomics_t params = {0};
    params.cmd_num = cmd_num;
    size_t byte_size = 0;

    for (int i = 0; i < cmd_num; i++) {
      params.command_length[i] = ((int *)command_length)[i];
      auto command_type = ((int *)_command_type)[i];
      params.command_type[i] = command_type;
      char *cmd = ((char *)command) + byte_size;

      char *pcmd = params.data + byte_size;
      for (int j = 0; j < params.command_length[i]; j++) {
        pcmd[j] = cmd[j];
      }

      byte_size += params.command_length[i];
    }
    auto func_id = tpu_kernel_get_function(bm_handle, tpu_module,
                                           "tpu_kernel_launch_atomics");
    auto ret = tpu_kernel_launch(bm_handle, func_id, &params, sizeof(params));

    return (int)ret;
  }

  void tpu_kernel_atomic_example() {
    bm_device_mem_t data_mem, weight_mem, output_mem;

    weight_mem = bm_mem_from_device(4294967296, 4 * sizeof(float));
    data_mem = bm_mem_from_device(4294971392, 16 * sizeof(float));
    output_mem = bm_mem_from_device(4294975488, 9 * sizeof(float));

    float data[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    float weight[4] = {20, 1, 2, 3};
    float output_data[9] = {0};

    bm_memcpy_s2d(bm_handle, weight_mem, (void *)weight);
    bm_memcpy_s2d(bm_handle, data_mem, (void *)data);

    sg_api_atomic_t params = {0};
    func_id = tpu_kernel_get_function(bm_handle, tpu_module,
                                      "tpu_kernel_launch_atomic_example");
    auto lret = tpu_kernel_launch(bm_handle, func_id, &params, sizeof(params));

    ASSERT_SUCCESS({bm_memcpy_d2s(bm_handle, (void *)output_data, output_mem)});
    std::cout << func_id << std::endl;
    for (int i = 0; i < 9; i++) {
      std::cout << output_data[i] << ',';
    }
    std::cout << std::endl;
    std::cout << "launch" << lret << std::endl;
  }

  // private:
  bm_handle_t bm_handle;
  tpu_kernel_function_t func_id;
  u64 reserved_offset;
  tpu_kernel_module_t tpu_module;
  bm_device_mem_t device_mem;
};

#ifdef __cplusplus
extern "C" {
#endif

void *init_handle(const char *fn, int devid) {
  ChipRunner *runner = new ChipRunner(fn, devid);
  return runner;
}

void *init_handle_b(const char *data, size_t length, int devid) {
  ChipRunner *runner = new ChipRunner(data, length, devid);
  return runner;
}

int chip_s2d(void *runner, u64 address, size_t size, void *data) {
  return ((ChipRunner *)runner)->chip_s2d(address, size, data);
}
int chip_d2s(void *runner, u64 address, size_t size, void *data) {
  return ((ChipRunner *)runner)->chip_d2s(address, size, data);
}
int launch_single_cmd(void *runner, void *command, int command_type,
                      size_t size) {
  return ((ChipRunner *)runner)->launch_single_cmd(command, command_type, size);
}
int launch_cmds(void *runner, void *command, void *_command_type,
                void *command_length, int cmd_num) {
  return ((ChipRunner *)runner)
      ->launch_cmds(command, _command_type, command_length, cmd_num);
}

u64 get_reserved_mem(void *runner) {
  return ((ChipRunner *)runner)->get_reserved_mem();
}

// BM1684X
#define GLOBAL_MEM_START_ADDR 0x100000000



/**
 * adpted from tpu-runtime/src/bmruntime.cpp:491@convert_cmd
 * currently only support BM1684X
*/
void convert_addr(u32 *cmd, u64 reserved_offset) {

  u64 src_addr = ((u64)(cmd[17] & 0xff) << 32) | ((u64)cmd[16]);
  u64 dst_addr = ((u64)(cmd[19] & 0xff) << 32) | ((u64)cmd[18]);
  bool src_in_global = src_addr >= GLOBAL_MEM_START_ADDR;
  bool dst_in_global = dst_addr >= GLOBAL_MEM_START_ADDR;
  u64 fix_addr;
  if (src_in_global) {
    fix_addr = src_addr + reserved_offset;
    if (fix_addr != src_addr) {
      cmd[16] = fix_addr & 0xffffffff;
      cmd[17] = ((u32)((fix_addr >> 32) & 0xff)) | (cmd[17] & 0xffffff00);
    }
  }
  if (dst_in_global) {
    fix_addr = dst_addr + reserved_offset;
    if (fix_addr != dst_addr) {
      cmd[18] = fix_addr & 0xffffffff;
      cmd[19] = ((u32)((fix_addr >> 32) & 0xff)) | (cmd[19] & 0xffffff00);
    }
  }
  // cmd type: 0:DMA_tensor, 1:DMA_matrix, 2:DMA_masked_select, 3:DMA_general
  // 4:DMA_cw_trans, 5:DMA_nonzero, 6:DMA_sys, 7:DMA_gather, 8:DMA_scatter
  // fix index_tensor or mask_tensor addr
  int cmd_type = (cmd[1] & 0x0f);
  if (cmd_type == 2 || cmd_type == 7 || cmd_type == 8) {
    u64 index_addr = ((u64)(cmd[21] & 0xff) << 32) | ((u64)cmd[20]);
    if (index_addr >= GLOBAL_MEM_START_ADDR) {
      fix_addr = index_addr + reserved_offset;
      if (fix_addr != index_addr) {
        cmd[20] = fix_addr & 0xffffffff;
        cmd[21] = ((u32)((fix_addr >> 32) & 0xff)) | (cmd[21] & 0xffffff00);
      }
    }
  }
}

void deinit(void *runner) { delete (ChipRunner *)runner; }

#ifdef __cplusplus
}
#endif

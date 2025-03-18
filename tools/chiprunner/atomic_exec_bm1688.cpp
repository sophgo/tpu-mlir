//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "bmlib_runtime.h"
#include "tpu_kernel.h"
#include <cassert>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>

// BM1684X
#define BM1684X_GLOBAL_MEM_START_ADDR 0x100000000
#define MAX_CMD_NUM 4

typedef struct {
  u64 global_addr;
  int shape[4];
  int dst_stride[4];
  int src_stride[4];
  int dtype; // data_type_t
#ifndef WIN32
} __attribute__((packed)) sg_api_debug_lmem_t;
#else
} sg_api_debug_lmem_t;
#endif

typedef unsigned long long u64;
typedef unsigned int u32;

#define ASSERT_SUCCESS(command)                                                \
  {                                                                            \
    bm_status_t status_;                                                       \
    status_ = command;                                                         \
    if (status_ != BM_SUCCESS) {                                               \
      std::cout << #command << "  " << __func__ << ":" << __LINE__             \
                << "   command failed" << std::endl;                           \
      exit(-1);                                                                \
    }                                                                          \
  }

#if 0
#define DEBUG_MEM() printf("Device mem %lu\n", device_mem.u.device.device_addr)
#else
#define DEBUG_MEM()
#endif

enum MEM_TAG {
  COEFF_TAG = 1,
  NEURON_TAG = 2,
  TIUBUF_TAG = 8,
  GDMABUF_TAG = 9,
  S2L_TAG = 10,
  L2S_TAG = 11,
};

#define TAG_SIZE 14

#define MAX_KERNEL_NUM 2

#define ID_BM1684X 0x1686
#define ID_BM1688 0x1688

inline int32_t div_up(int32_t v, int32_t align) {
  return (v + align - 1) / align;
}

inline int32_t align_up(int32_t v, int32_t align) {
  return div_up(v, align) * align;
}
class DeviceRunner {

public:
  DeviceRunner(int device_flag, int dev_id) {
    if (device_flag == ID_BM1684X) {
      localmem_size = (1 << 14) // BANK_SIZE
                      * 16      // BANK_PER_LANE
                      * 64      // NPU_NUM
          ;
    } else if (device_flag == ID_BM1688) {
      localmem_size = (1 << 17) // LANE_SIZE
                      * 32      // NPU_NUM
          ;
    } else {
      assert(0);
    }

    ASSERT_SUCCESS(bm_dev_request(&bm_handle, dev_id));
    ASSERT_SUCCESS(bm_get_tpu_core_num(bm_handle, &max_core_num));

    device_flag_ = device_flag;
    memset(mallocate_memorys, 0, TAG_SIZE * sizeof(bm_device_mem_t));
  }
  DeviceRunner(const char *fn, int device_flag, int devid = 0)
      : DeviceRunner(device_flag, devid) {

    for (int i = 0; i < max_core_num; i++) {
      tpu_module[i] = tpu_kernel_load_module_file_to_core(bm_handle, fn, i);
    }
    printf("load %s", fn);
  }

  DeviceRunner(const char *data, size_t length, int device_flag, int devid = 0)
      : DeviceRunner(device_flag, devid) {
    tpu_kernel_load_module(bm_handle, data, length);
  }

  u64 init_memory(int tag, u64 size) {
    if (mallocate_memorys[tag].u.device.device_addr != 0) {
      printf("Free %lu\n", mallocate_memorys[tag].u.device.device_addr);
      bm_free_device(bm_handle, mallocate_memorys[tag]);
    }
    auto status = bm_malloc_device_byte_heap_mask(
        bm_handle, &mallocate_memorys[tag], 7, size);
    printf("Allocate %lu for %llu \n",
           mallocate_memorys[tag].u.device.device_addr, size);
    ASSERT_SUCCESS(status);
    return mallocate_memorys[tag].u.device.device_addr;
  }

  u64 addr_from_offset(u64 offset, int tag = -1) {
    if (tag >= 0) {
      auto memory = mallocate_memorys[tag];
      assert(memory.u.device.device_addr != 0);
      return memory.u.device.device_addr + offset;
    }
    return offset;
  }
  u64 offset_from_tag(int tag = -1) {
    if (tag >= 0) {
      auto memory = mallocate_memorys[tag];
      assert(memory.u.device.device_addr != 0);
      if (device_flag_ == ID_BM1684X) {
        return memory.u.device.device_addr - (1UL << 32);
      } else {
        return memory.u.device.device_addr - (1UL << 36);
      }
    }
    return 0;
  }

  bm_device_mem_t mem_from_offset(u64 offset, size_t size, int tag) {
    auto target_addr = addr_from_offset(offset, tag);
    bm_device_mem_t device_mem = bm_mem_from_device(target_addr, size);
    return device_mem;
  }

  void memcpy_cmd(bm_device_mem_t device_mem, void *data) {}

  void memcpy_s2d(u64 addr, size_t size, void *data, int tag = -1) {
    auto dmem = mem_from_offset(addr, size, tag);
    memcpy_s2d(dmem, data);
  }

  void memcpy_d2s(u64 addr, size_t size, void *data, int tag = -1) {
    auto dmem = mem_from_offset(addr, size, tag);
    memcpy_d2s(dmem, data);
  }

  void memcpy_l2s(void *data, u32 core_id = 0) {
    auto dmem = mem_from_offset(0, localmem_size, L2S_TAG);
    memcpy_l2s(dmem, data, core_id);
  }

  void memcpy_s2l(void *data, u32 core_id = 0) {
    auto dmem = mem_from_offset(0, localmem_size, S2L_TAG);
    memcpy_s2l(dmem, data, core_id);
  }

  void memcpy_s2d(bm_device_mem_t device_mem, void *data) {
    auto ret = bm_memcpy_s2d(bm_handle, device_mem, (void *)data);
    DEBUG_MEM();
    ASSERT_SUCCESS(ret);
  }
  void memcpy_d2s(bm_device_mem_t device_mem, void *data) {
    auto ret = bm_memcpy_d2s(bm_handle, (void *)data, device_mem);
    DEBUG_MEM();
    ASSERT_SUCCESS(ret);
  }
  void memcpy_s2l(bm_device_mem_t device_mem, void *data, u32 core_id = 0) {
    assert(core_id < max_core_num);
    auto func_id = tpu_kernel_get_function_from_core(
        bm_handle, tpu_module[core_id], "tpu_kernel_gdma_cpy_s2l", core_id);

    memcpy_s2d(device_mem, data);

    sg_api_debug_lmem_t params = {0};
    params.global_addr = device_mem.u.device.device_addr;
    if (device_flag_ == ID_BM1684X) {
      int shape[4] = {1, 64, 512, 128};
      memcpy(params.shape, shape, sizeof(shape));
    } else if (device_flag_ == ID_BM1688) {
      int shape[4] = {1, 32, 512, 64};
      memcpy(params.shape, shape, sizeof(shape));
    }
    params.dtype = DT_UINT32;

    auto status = tpu_kernel_launch_from_core(bm_handle, func_id, &params,
                                              sizeof(params), core_id);
    ASSERT_SUCCESS(status);
    // ASSERT_SUCCESS(tpu_kernel_sync(bm_handle));
  }

  void memcpy_l2s(bm_device_mem_t device_mem, void *data, u32 core_id = 0) {
    assert(core_id < max_core_num);
    auto func_id = tpu_kernel_get_function_from_core(
        bm_handle, tpu_module[core_id], "tpu_kernel_gdma_cpy_l2s", core_id);

    sg_api_debug_lmem_t params = {0};
    params.global_addr = device_mem.u.device.device_addr;
    if (device_flag_ == ID_BM1684X) {
      int shape[4] = {1, 64, 512, 128};
      memcpy(params.shape, shape, sizeof(shape));
    } else if (device_flag_ == ID_BM1688) {
      int shape[4] = {1, 32, 512, 64};
      memcpy(params.shape, shape, sizeof(shape));
    }

    params.dtype = DT_UINT32;

    auto status = tpu_kernel_launch_from_core(bm_handle, func_id, &params,
                                              sizeof(params), core_id);
    ASSERT_SUCCESS(status);
    // ASSERT_SUCCESS(tpu_kernel_sync(bm_handle));

    memcpy_d2s(device_mem, data);
  }

  ~DeviceRunner() {
    for (int i = 0; i < MAX_KERNEL_NUM; i++) {
      if (tpu_module[i]) {
        tpu_kernel_unload_module_from_core(
            bm_handle, (tpu_kernel_module_t)tpu_module[i], i);
        tpu_module[i] = 0;
      }
    }
    if (bm_handle) {
      // bm_free_device(bm_handle, device_mem);
      // bm_free_device(bm_handle, local_mem_buffer);
      // bm_free_device(bm_handle, tiu_mem);
      // bm_free_device(bm_handle, dma_mem);
      for (int i = 0; i < TAG_SIZE; i++) {
        if (mallocate_memorys[i].u.device.device_addr == 0) {
          continue;
        }
        bm_free_device(bm_handle, mallocate_memorys[i]);
      }
      bm_dev_free(bm_handle);
    }
  }

  void debug_cmds(u64 *tiu_cmds, u64 *dma_cmds, size_t tiu_buf_len,
                  size_t dma_buf_len, int tiu_nums, int dma_nums,
                  u32 core_id = 0) {
    assert(core_id < max_core_num);

    u64 tiubuf_addr = addr_from_offset(0, TIUBUF_TAG);
    u64 dmabuf_addr = addr_from_offset(0, GDMABUF_TAG);
    ASSERT_SUCCESS(bm_memcpy_s2d_partial(
                       bm_handle, mem_from_offset(0, tiu_buf_len, TIUBUF_TAG),
                       (void *)tiu_cmds, (u32)tiu_buf_len);)
    ASSERT_SUCCESS(bm_memcpy_s2d_partial(
        bm_handle, mem_from_offset(0, dma_buf_len, GDMABUF_TAG),
        (void *)dma_cmds, (u32)dma_buf_len));

    size_t total_size = sizeof(u64) * 3;
    u64 *p64_cmd = (u64 *)malloc(total_size);
    u32 *p32_cmd = (u32 *)p64_cmd;
    p32_cmd[0] = tiu_nums;
    p32_cmd[1] = dma_nums;
    p64_cmd[1] = tiubuf_addr;
    p64_cmd[2] = dmabuf_addr;

    tpu_kernel_function_t func_id = tpu_kernel_get_function_from_core(
        bm_handle, tpu_module[core_id], "tpu_kernel_debug_cmds", core_id);

    auto ret = tpu_kernel_launch_from_core(bm_handle, func_id, p64_cmd,
                                           total_size, core_id);
    ASSERT_SUCCESS(ret);
    free(p64_cmd);
  }

  /**
   * adpted from tpu-runtime/src/bmruntime.cpp:491@convert_cmd
   * currently only support BM1684X
   */
  void convert_addr(u32 *cmd) {
    if (device_flag_ == ID_BM1684X) {
      u64 neuron_offset = addr_from_offset(0, NEURON_TAG);
      u64 src_addr = ((u64)(cmd[17] & 0xff) << 32) | ((u64)cmd[16]);
      u64 dst_addr = ((u64)(cmd[19] & 0xff) << 32) | ((u64)cmd[18]);
      bool src_in_global = src_addr >= BM1684X_GLOBAL_MEM_START_ADDR;
      bool dst_in_global = dst_addr >= BM1684X_GLOBAL_MEM_START_ADDR;
      u64 fix_addr;
      if (src_in_global) {
        fix_addr = src_addr + neuron_offset;
        if (fix_addr != src_addr) {
          cmd[16] = fix_addr & 0xffffffff;
          cmd[17] = ((u32)((fix_addr >> 32) & 0xff)) | (cmd[17] & 0xffffff00);
        }
      }
      if (dst_in_global) {
        fix_addr = dst_addr + neuron_offset;
        if (fix_addr != dst_addr) {
          cmd[18] = fix_addr & 0xffffffff;
          cmd[19] = ((u32)((fix_addr >> 32) & 0xff)) | (cmd[19] & 0xffffff00);
        }
      }
      // cmd type: 0:DMA_tensor, 1:DMA_matrix, 2:DMA_masked_select,
      // 3:DMA_general 4:DMA_cw_trans, 5:DMA_nonzero, 6:DMA_sys, 7:DMA_gather,
      // 8:DMA_scatter fix index_tensor or mask_tensor addr
      int cmd_type = (cmd[1] & 0x0f);
      if (cmd_type == 2 || cmd_type == 7 || cmd_type == 8) {
        u64 index_addr = ((u64)(cmd[21] & 0xff) << 32) | ((u64)cmd[20]);
        if (index_addr >= BM1684X_GLOBAL_MEM_START_ADDR) {
          fix_addr = index_addr + neuron_offset;
          if (fix_addr != index_addr) {
            cmd[20] = fix_addr & 0xffffffff;
            cmd[21] = ((u32)((fix_addr >> 32) & 0xff)) | (cmd[21] & 0xffffff00);
          }
        }
      }
    } else if (device_flag_ == ID_BM1688) {

      int cmd_type = (cmd[1] & 0x0f);
      if (cmd_type == 6)
        return; // cmd_type: DMA_sys
      u64 src_addr = ((u64)(cmd[17] & 0xff) << 32) | ((u64)cmd[16]);
      u64 dst_addr = ((u64)(cmd[19] & 0xff) << 32) | ((u64)cmd[18]);
      bool src_in_global = (src_addr >> 39) & 0x1;
      bool dst_in_global = (dst_addr >> 39) & 0x1;
      u64 fix_addr;
      if (src_in_global) {
        int tag = ((src_addr >> 36) & 0x7);
        u64 tag_offset = tag > 0 ? addr_from_offset(0, tag)
                                 : addr_from_offset(0, NEURON_TAG);

        fix_addr = ((src_addr & ((1ull << 35) - 1))) + tag_offset;
        fix_addr |= (1ull << 39);
        if (fix_addr != src_addr) {
          cmd[16] = fix_addr & 0xffffffff;
          cmd[17] = ((u32)((fix_addr >> 32) & 0xff)) | (cmd[17] & 0xffffff00);
        }
      }
      if (dst_in_global) {
        int tag = ((dst_addr >> 36) & 0x7);
        u64 tag_offset = tag > 0 ? addr_from_offset(0, tag)
                                 : addr_from_offset(0, NEURON_TAG);

        fix_addr = (dst_addr & ((1ull << 35) - 1)) + tag_offset;
        fix_addr |= (1ull << 39);
        if (fix_addr != dst_addr) {
          cmd[18] = fix_addr & 0xffffffff;
          cmd[19] = ((u32)((fix_addr >> 32) & 0xff)) | (cmd[19] & 0xffffff00);
        }
      }
      // cmd type: 0:DMA_tensor, 1:DMA_matrix, 2:DMA_masked_select,
      // 3:DMA_general 4:DMA_cw_trans, 5:DMA_nonzero, 6:DMA_sys, 7:DMA_gather,
      // 8:DMA_scatter 9:DMA_reverse 10:DMA_compress 11: DMA_decompress fix
      // index_tensor or mask_tensor addr
      if (cmd_type == 2 || cmd_type == 7 || cmd_type == 8 || cmd_type == 0xa ||
          cmd_type == 0xb) {
        u64 index_addr = ((u64)(cmd[21] & 0xff) << 32) | ((u64)cmd[20]);
        if (((index_addr >> 39) & 0x1)) {
          int tag = ((index_addr >> 36) & 0x7);
          u64 tag_offset = tag > 0 ? addr_from_offset(0, tag)
                                   : addr_from_offset(0, NEURON_TAG);

          fix_addr = (index_addr & ((1ull << 35) - 1)) + tag_offset;
          fix_addr |= (1ull << 39);
          if (fix_addr != index_addr) {
            cmd[20] = fix_addr & 0xffffffff;
            cmd[21] = ((u32)((fix_addr >> 32) & 0xff)) | (cmd[21] & 0xffffff00);
          }
        }
      }
    }
  }

  // private:
  u32 max_core_num = 1;
  size_t localmem_size = 0;
  int device_flag_ = 0;
  bm_handle_t bm_handle;
  tpu_kernel_function_t func_id;
  tpu_kernel_module_t tpu_module[MAX_KERNEL_NUM] = {0};
  bm_device_mem_t *mallocate_memorys = new bm_device_mem_t[TAG_SIZE];
};

#ifdef __cplusplus
extern "C" {
#endif

void *init_handle(const char *fn, int devid, int device_flag) {
  DeviceRunner *runner = new DeviceRunner(fn, device_flag, devid);
  return runner;
}

void *init_handle_b(const char *data, size_t length, int devid,
                    int device_flag) {
  DeviceRunner *runner = new DeviceRunner(data, length, device_flag, devid);
  return runner;
}

void memcpy_s2d(void *runner, u64 address, size_t size, void *data,
                int tag = -1) {
  return ((DeviceRunner *)runner)->memcpy_s2d(address, size, data, tag);
}
void memcpy_d2s(void *runner, u64 address, size_t size, void *data,
                int tag = -1) {
  return ((DeviceRunner *)runner)->memcpy_d2s(address, size, data, tag);
}
void memcpy_l2s(void *runner, void *data, u32 core_id = 0) {
  return ((DeviceRunner *)runner)->memcpy_l2s(data, core_id);
}

void memcpy_s2l(void *runner, void *data, u32 core_id = 0) {
  return ((DeviceRunner *)runner)->memcpy_s2l(data, core_id);
}

u64 init_memory(void *runner, int tag, u32 size) {
  return ((DeviceRunner *)runner)->init_memory(tag, size);
}

void debug_cmds(void *runner, u64 *tiu_cmds, u64 *dma_cmds, size_t tiu_buf_len,
                size_t dma_buf_len, int tiu_nums, int dma_nums,
                u32 core_id = 0) {
  ((DeviceRunner *)runner)
      ->debug_cmds(tiu_cmds, dma_cmds, tiu_buf_len, dma_buf_len, tiu_nums,
                   dma_nums, core_id);
}

void convert_addr(void *runner, u32 *cmd) {
  ((DeviceRunner *)runner)->convert_addr(cmd);
}

u32 get_max_core_num(void *runner) {
  return ((DeviceRunner *)runner)->max_core_num;
}

// u64 get_reserved_mem(void *runner) {
//   return ((DeviceRunner *)runner)->get_reserved_mem();
// }

void deinit(void *runner) { delete (DeviceRunner *)runner; }

#ifdef __cplusplus
}
#endif

#include <cstdio>
#include <cstdlib> // For malloc and free

int main(int argc, char **argv) {
  int *ori_data =
      (int *)malloc(1 * 1024 * 1024 * 16 * sizeof(int)); // 动态分配内存
  for (int i = 0; i < 100; i++) {
    ori_data[i] = i;
  }
  int *new_data =
      (int *)malloc(1 * 1024 * 1024 * 16 * sizeof(int)); // 动态分配内存
  int *new_data_ddr =
      (int *)malloc(1 * 1024 * 1024 * 16 * sizeof(int)); // 动态分配内存
  if (new_data == nullptr) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1; // 处理内存分配失败
  }

  auto runner = DeviceRunner(
      "/workspace/tpu-mlir/install/lib/libbmtpulv60_kernel_module.so",
      ID_BM1688, 0);
  runner.init_memory(S2L_TAG, 16 * 1024 * 1024);
  runner.init_memory(L2S_TAG, 16 * 1024 * 1024);
  runner.memcpy_s2l(ori_data);
  runner.memcpy_d2s(0, 100, new_data_ddr, S2L_TAG);
  runner.memcpy_l2s(new_data);

  free(new_data); // 释放动态分配的内存
  printf("\n");
  return 0;
}

/*****************************************************************************
 *
 *    Copyright (c) 2016-2026 by Bitmain Technologies Inc. All rights reserved.
 *
 *    The material in this file is confidential and contains trade secrets
 *    of Bitmain Technologies Inc. This is proprietary information owned by
 *    Bitmain Technologies Inc. No part of this work may be disclosed,
 *    reproduced, copied, transmitted, or used in any way for any purpose,
 *    without the express written permission of Bitmain Technologies Inc.
 *
 *****************************************************************************/

/**************************************************************************
 * bmlib_runtime defines interfaces that operate TPU devices.
 * The functions can be divided into serveral categories.
 * 1) device handle creation and destroy
 * 2) memory help functions
 * 3) global memory allocation and free
 * 4) data transfer between host and device
 * 5) data transfer within device memory
 * 6) api send and synchronization
 * 7) global memory map and coherence
 * 8) trace and profile
 * 9) power management
 * 10) miscellaneous functions
 *************************************************************************/

#ifndef BMLIB_RUNTIME_H_
#define BMLIB_RUNTIME_H_
#if defined(_WIN32) && !defined(__MINGW32__)
    #include <vadefs.h>
    #define DECL_EXPORT __declspec(dllexport)
    #define DECL_IMPORT __declspec(dllimport)
#else
	#include <stdbool.h>
	#include <stddef.h>
	#include <stdarg.h>
    #define DECL_EXPORT
    #define DECL_IMPORT
#endif

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum {
  MODULE_CDMA = 0,
  MODULE_GDMA = 1,
  MODULE_TPU = 2,
  MODULE_SMMU = 3,
  MODULE_SRAM = 4,
  MODULE_END = 5
} MODULE_ID;

#define BM_MEM_ADDR_NULL (0xfffffffff)

#ifndef BM_MEM_DESC_T_
#define BM_MEM_DESC_T_
/* BM function return code definitions */
typedef enum {
  BM_SUCCESS = 0,
  BM_ERR_DEVNOTREADY = 1, /* Device not ready yet */
  BM_ERR_FAILURE = 2,     /* General failure */
  BM_ERR_TIMEOUT = 3,     /* Timeout */
  BM_ERR_PARAM = 4,       /* Parameters invalid */
  BM_ERR_NOMEM = 5,       /* Not enough memory */
  BM_ERR_DATA = 6,        /* Data error */
  BM_ERR_BUSY = 7,        /* Busy */
  BM_ERR_NOFEATURE = 8,   /* Not supported yet */
  BM_NOT_SUPPORTED = 9
} bm_status_t;

/* BM memory type definitions */
typedef enum {
  BM_MEM_TYPE_DEVICE = 0,
  BM_MEM_TYPE_HOST = 1,
  BM_MEM_TYPE_SYSTEM = 2,
  BM_MEM_TYPE_INT8_DEVICE = 3,
  BM_MEM_TYPE_INVALID = 4
} bm_mem_type_t;

typedef enum {
  PERF_MONITOR_GDMA = 0,
  PERF_MONITOR_TPU = 1
} PERF_MONITOR_ID;

typedef enum {
  BMCPU_IDLE    = 0,
  BMCPU_RUNNING = 1,
  BMCPU_FAULT   = 2
} bm_cpu_status_t;

/*
* bm performace monitor
*/
typedef struct bm_perf_monitor {
  long long buffer_start_addr; /*buffer address to store perf data*/
  int buffer_size; /*buffer size*/
  PERF_MONITOR_ID monitor_id; /*PERF_MONITOR_GDMA or PERF_MONITOR_TPU*/
} bm_perf_monitor_t;

typedef union {
  struct {
    bm_mem_type_t mem_type : 3;
    unsigned int gmem_heapid : 3;
    unsigned int reserved : 26;
  } u;
  unsigned int rawflags;
} bm_mem_flags_t;

/* BM memory descriptor definition*/
typedef struct bm_mem_desc {
  union {
    struct {
#ifdef __linux__
      unsigned long device_addr;
#else
      unsigned long long device_addr;
#endif
      unsigned int reserved;
      int dmabuf_fd;
    } device;

    struct {
      void *system_addr;
      unsigned int reserved0;
      int reserved1;
    } system;
  } u;

  bm_mem_flags_t flags;
  unsigned int size;
} bm_mem_desc_t;

typedef struct bm_mem_desc bm_device_mem_t;
typedef struct bm_mem_desc bm_system_mem_t;

typedef struct sg_mem_desc {
  union {
    struct {
#ifdef __linux__
      unsigned long device_addr;
#else
      unsigned long long device_addr;
#endif
      unsigned int reserved;
      int dmabuf_fd;
    } device;

    struct {
      void *system_addr;
      unsigned int reserved0;
      int reserved1;
    } system;
  } u;

  bm_mem_flags_t flags;
  unsigned long long size;
} sg_mem_desc_t;

typedef struct sg_mem_desc sg_device_mem_t;
typedef struct sg_mem_desc sg_system_mem_t;

typedef struct bm_mem_desc_u64 {
  union {
    struct {
#ifdef __linux__
      unsigned long device_addr;
#else
      unsigned long long device_addr;
#endif
      unsigned int reserved;
      int dmabuf_fd;
    } device;

    struct {
      void *system_addr;
      unsigned int reserved0;
      int reserved1;
    } system;
  } u;

  bm_mem_flags_t flags;
  unsigned long long size;
} bm_mem_desc_u64_t;

typedef struct bm_mem_desc_u64 bm_device_mem_u64_t;
typedef struct bm_mem_desc_u64 bm_system_mem_u64_t;
#endif

struct bm_context;
typedef struct bm_context *bm_handle_t;

#define MD5SUM_LEN 16
#define LIB_MAX_NAME_LEN 64
#define FUNC_MAX_NAME_LEN 64

typedef struct bm_module
{
  // void *lib_handle;
  char lib_name[LIB_MAX_NAME_LEN];
  unsigned char md5[MD5SUM_LEN];
}bm_module;

typedef struct bm_module *tpu_kernel_module_t;
typedef int tpu_kernel_function_t;

/**
 * @name    tpu_kernel_load_module_file
 * @brief   To load dyn file
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  module_file     dyn file
 * @retval  dyn lib ptr
 */
tpu_kernel_module_t tpu_kernel_load_module_file(bm_handle_t handle, const char *module_file);

/**
 * @name    tpu_kernel_load_module_file_to_core
 * @brief   To load dyn file
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  module_file     dyn file
 * @param [in]  core_id
 * @retval  dyn lib ptr
 */
tpu_kernel_module_t tpu_kernel_load_module_file_to_core(bm_handle_t handle, const char *module_file, int core_id);

/**
 * @name    tpu_kernel_load_module_file_key
 * @brief   To load dyn file with key
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  module_file     dyn file
 * @param [in]  key             identification str
 * @param [in]  size            key size
 * @retval  dyn lib ptr
 */
tpu_kernel_module_t tpu_kernel_load_module_file_key(bm_handle_t handle, const char *module_file, const char *key, int size);

/**
 * @name    tpu_kernel_unload_module
 * @brief   To unload dyn file
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  p_module        dyn lib ptr
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t tpu_kernel_unload_module(bm_handle_t handle, tpu_kernel_module_t p_module);

/**
 * @name    tpu_kernel_unload_module_from_core
 * @brief   To unload dyn file
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  p_module        dyn lib ptr
 * @param [in]  core_id         core id
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t tpu_kernel_unload_module_from_core(bm_handle_t handle, tpu_kernel_module_t p_module, int core_id);

/**
 * @name    tpu_kernel_free_module
 * @brief   To free p_module when not use
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  p_module        dyn lib ptr
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t tpu_kernel_free_module(bm_handle_t handle, tpu_kernel_module_t p_module);

/**
 * @name    tpu_kernel_load_module
 * @brief   To load dyn module
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  data            dyn module
 * @param [in]  length          dyn module size
 * @retval  dyn lib ptr
 */
tpu_kernel_module_t tpu_kernel_load_module(bm_handle_t handle, const char *data, size_t length);

/**
 * @name    tpu_kernel_load_module_to_core
 * @brief   To load dyn module
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  data            dyn module
 * @param [in]  length          dyn module size
 * @param [in]  core_id         core id
 * @retval  dyn lib ptr
 */
tpu_kernel_module_t tpu_kernel_load_module_to_core(bm_handle_t handle, const char *data, size_t length, int core_id);

/**
 * @name    tpu_kernel_get_function
 * @brief   To get function from lib
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  module          dyn module
 * @param [in]  function        funtion name
 * @retval  function id
 */
tpu_kernel_function_t tpu_kernel_get_function(bm_handle_t handle, tpu_kernel_module_t module, const char *function);

/**
 * @name    tpu_kernel_get_function_from_core
 * @brief   To get function from lib
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  module          dyn module
 * @param [in]  function        funtion name
 * @param [in]  core_id         core id
 * @retval  function id
 */
tpu_kernel_function_t tpu_kernel_get_function_from_core(bm_handle_t handle, tpu_kernel_module_t module, const char *function, int core_id);

/**
 * @name    tpu_kernel_launch
 * @brief   To launch function with sync
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  function        function id
 * @param [in]  args            funtion args
 * @param [in]  size            args size
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t tpu_kernel_launch(bm_handle_t handle, tpu_kernel_function_t function, void *args, size_t size);

/**
 * @name    tpu_kernel_launch_from_core
 * @brief   To launch function with sync
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  function        function id
 * @param [in]  args            funtion args
 * @param [in]  size            args size
 * @param [in]  core_id         core id
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t tpu_kernel_launch_from_core(bm_handle_t handle, tpu_kernel_function_t function, void *args, size_t size, int core_id);

/**
 * @name    tpu_kernel_launch_async
 * @brief   To launch function with async
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  function        function id
 * @param [in]  args            funtion args
 * @param [in]  size            args size
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t tpu_kernel_launch_async(bm_handle_t handle, tpu_kernel_function_t function, void *args, size_t size);

/**
 * @name    tpu_kernel_launch_async_from_core
 * @brief   To launch function with async
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  function        function id
 * @param [in]  args            funtion args
 * @param [in]  size            args size
 * @param [in]  core_id         core_id
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t tpu_kernel_launch_async_from_core(bm_handle_t handle, tpu_kernel_function_t function, void *args, size_t size, int core_id);

/**
 * @name    tpu_kernel_launch_async_multi_cores
 * @brief   To launch function with async for multi cores
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  func_name       function name
 * @param [in]  api_param       funtion params
 * @param [in]  api_size        params size
 * @param [in]  core_list       list of core ids
 * @param [in]  core_num        number of cores
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t tpu_kernel_launch_async_multi_cores(bm_handle_t handle, const char *func_name, const void *api_param,
                                                size_t api_size, const int* core_list, const int core_num);

/**
 * @name    tpu_kernel_launch_sync_multi_cores
 * @brief   To launch function with sync for multi cores
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  func_name       function name
 * @param [in]  api_param       funtion params
 * @param [in]  api_size        params size
 * @param [in]  core_list       list of core ids
 * @param [in]  core_num        number of cores
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t tpu_kernel_launch_sync_multi_cores(bm_handle_t handle, const char *func_name, const void *api_param,
                                              size_t api_size, const int* core_list, const int core_num);

/**
 * @name    tpu_kernel_sync
 * @brief   To sync
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t tpu_kernel_sync(bm_handle_t handle);
void show_md5(unsigned char md5[]);

DECL_EXPORT void bmlib_log(const char *tag, int level, const char *fmt, ...);

#ifndef USING_CMODEL
#define BM_CHECK_RET(call)                                                    \
  do {                                                                        \
    bm_status_t ret = (bm_status_t)call;                                                   \
    if (ret != BM_SUCCESS) {                                                  \
      bmlib_log("BM_CHECK",16,"BM_CHECK_RET fail %s: %s: %d\n", __FILE__, __func__, __LINE__); \
      return ret;                                                             \
    }                                                                         \
  } while (0)
#else
#define BM_CHECK_RET(call)                     \
  do {                                         \
    bm_status_t ret = call;                    \
    if (ret != BM_SUCCESS) {                   \
      bmlib_log("BM_CHECK",16,"BM_CHECK_RET failed %d\n", ret);\
      ASSERT(0);                               \
      exit(-ret);                              \
    }                                          \
  } while (0)
#endif

/*******************handle releated functions *********************************/
/**
 * @name    bm_dev_getcount
 * @brief   To get the number of sophon devices in system.
 *          If N is got, valid devid is [0, N-1]
 * @ingroup bmlib_runtime
 *
 * @param [out] count  The result number of sophon devices
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_dev_getcount(int *count);

/**
 * @name    bm_dev_query
 * @brief   To query if a device is present
 * @ingroup bmlib_runtime
 *
 * @param [in] devid  The id of the device to query
 * @retval  BM_SUCCESS Device is present
 *          Other code Devcie is not present
 */
DECL_EXPORT bm_status_t bm_dev_query(int devid);

/**
 * @name    bm_dev_request
 * @brief   To create a handle for the given device
 * @ingroup bmlib_runtime
 *
 * @param [out] handle  The created handle
 * @param [in]  devid   Specify on which device to create handle
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_dev_request(bm_handle_t *handle, int devid);

/**
 * @name    bm_get_devid
 * @brief   To get device index for the given handle
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The given handle
 * @retval  int  device index that the handle points to.
 */
DECL_EXPORT int bm_get_devid(bm_handle_t handle);

/**
 * @name    bm_dev_free
 * @brief   To free a handle
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The handle to free
 */
DECL_EXPORT void bm_dev_free(bm_handle_t handle);

/*******************memory help functions ************************************/
/**
 * @name    bm_mem_get_type
 * @brief   To get a memory descriptor's type
 * @ingroup bmlib_runtime
 *
 * @param [in]  mem  The memory descriptor queried
 * @retval  BM_MEM_TYPE_DEVICE  Device global memory
 * @retval  BM_MEM_TYPE_SYSTEM  Host user memory
 */
DECL_EXPORT bm_mem_type_t bm_mem_get_type(struct bm_mem_desc mem);

/**
 * @name    sg_mem_get_type
 * @brief   To get a memory descriptor's type
 * @ingroup bmlib_runtime
 *
 * @param [in]  mem  The memory descriptor queried
 * @retval  BM_MEM_TYPE_DEVICE  Device global memory
 * @retval  BM_MEM_TYPE_SYSTEM  Host user memory
 */
DECL_EXPORT bm_mem_type_t sg_mem_get_type(struct sg_mem_desc mem);

/**
 * @name    bm_mem_get_type_u64
 * @brief   To get a memory descriptor's type
 * @ingroup bmlib_runtime
 *
 * @param [in]  mem  The memory descriptor queried
 * @retval  BM_MEM_TYPE_DEVICE  Device global memory
 * @retval  BM_MEM_TYPE_SYSTEM  Host user memory
 */
DECL_EXPORT bm_mem_type_t bm_mem_get_type_u64(struct bm_mem_desc_u64 mem);

/**
 * @name    bm_mem_get_device_addr
 * @brief   To get a device memory descriptor's address
 * @ingroup bmlib_runtime
 *
 * @param [in]  mem  The device memory descriptor queried
 * @retval  unsigned long long  The device memory address
 */
DECL_EXPORT unsigned long long bm_mem_get_device_addr(struct bm_mem_desc mem);

/**
 * @name    sg_mem_get_device_addr
 * @brief   To get a device memory descriptor's address
 * @ingroup bmlib_runtime
 *
 * @param [in]  mem  The device memory descriptor queried
 * @retval  unsigned long long  The device memory address
 */
DECL_EXPORT unsigned long long sg_mem_get_device_addr(struct sg_mem_desc mem);

/**
 * @name    bm_mem_get_device_addr_u64
 * @brief   To get a device memory descriptor's address
 * @ingroup bmlib_runtime
 *
 * @param [in]  mem  The device memory descriptor queried
 * @retval  unsigned long long  The device memory address
 */
DECL_EXPORT unsigned long long bm_mem_get_device_addr_u64(struct bm_mem_desc_u64 mem);

/**
 * @name    bm_mem_set_device_addr
 * @brief   To set a device memory descriptor's address
 * @ingroup bmlib_runtime
 *
 * @param [in]  pmem   The device memory descriptor pointer
 * @param ]in]  addr  The new device address of the device memory
 */
DECL_EXPORT void bm_mem_set_device_addr(struct bm_mem_desc* pmem, unsigned long long addr);

/**
 * @name    sg_mem_set_device_addr
 * @brief   To set a device memory descriptor's address
 * @ingroup bmlib_runtime
 *
 * @param [in]  pmem   The device memory descriptor pointer
 * @param ]in]  addr  The new device address of the device memory
 */
DECL_EXPORT void sg_mem_set_device_addr(struct sg_mem_desc* pmem, unsigned long long addr);

/**
 * @name    bm_mem_set_device_addr_u64
 * @brief   To set a device memory descriptor's address
 * @ingroup bmlib_runtime
 *
 * @param [in]  pmem   The device memory descriptor pointer
 * @param ]in]  addr  The new device address of the device memory
 */
DECL_EXPORT void bm_mem_set_device_addr_u64(struct bm_mem_desc_u64* pmem, unsigned long long addr);

/**
 * @name    bm_mem_get_device_size
 * @brief   To get a device memory descriptor's size
 * @ingroup bmlib_runtime
 *
 * @param [in]  mem      The device memory descriptor queried
 * @retval unsigned int  The device memory's size in bytes
 */
DECL_EXPORT unsigned int bm_mem_get_device_size(struct bm_mem_desc mem);

/**
 * @name    sg_mem_get_device_size
 * @brief   To get a device memory descriptor's size
 * @ingroup bmlib_runtime
 *
 * @param [in]  mem      The device memory descriptor queried
 * @retval unsigned int  The device memory's size in bytes
 */
DECL_EXPORT unsigned long long sg_mem_get_device_size(struct sg_mem_desc mem);

/**
 * @name    bm_mem_get_device_size_u64
 * @brief   To get a device memory descriptor's size
 * @ingroup bmlib_runtime
 *
 * @param [in]  mem      The device memory descriptor queried
 * @retval unsigned int  The device memory's size in bytes
 */
DECL_EXPORT unsigned long long bm_mem_get_device_size_u64(struct bm_mem_desc_u64 mem);

/**
 * @name    bm_mem_set_device_size
 * @brief   To set a device memory descriptor's size
 * @ingroup bmlib_runtime
 *
 * @param [out]  pmem  The device memory descriptor pointer
 * @param [in]  size  The new device memory size (in bytes) of the device memory
 */
DECL_EXPORT void bm_mem_set_device_size(struct bm_mem_desc* pmem, unsigned int size);

/**
 * @name    sg_mem_set_device_size
 * @brief   To set a device memory descriptor's size
 * @ingroup bmlib_runtime
 *
 * @param [out]  pmem  The device memory descriptor pointer
 * @param [in]  size  The new device memory size (in bytes) of the device memory
 */
DECL_EXPORT void sg_mem_set_device_size(struct sg_mem_desc* pmem, unsigned long long size);

/**
 * @name    bm_mem_set_device_size_u64
 * @brief   To set a device memory descriptor's size
 * @ingroup bmlib_runtime
 *
 * @param [out]  pmem  The device memory descriptor pointer
 * @param [in]  size  The new device memory size (in bytes) of the device memory
 */
DECL_EXPORT void bm_mem_set_device_size_u64(struct bm_mem_desc_u64* pmem, unsigned long long size);

/**
 * @name    bm_set_device_mem
 * @brief   To fill in a device memory descriptor with size and address
 * @ingroup bmlib_runtime
 *
 * @param [in] pmem  The device memory descriptor pointer
 * @param [in]  size  The device memory descriptor's size
 * @param [in]  addr  The device memory descriptor's address
 */
DECL_EXPORT void bm_set_device_mem(bm_device_mem_t* pmem, unsigned int size,
                       unsigned long long addr);

/**
 * @name    sg_set_device_mem
 * @brief   To fill in a device memory descriptor with size and address
 * @ingroup bmlib_runtime
 *
 * @param [in] pmem  The device memory descriptor pointer
 * @param [in]  size  The device memory descriptor's size
 * @param [in]  addr  The device memory descriptor's address
 */
DECL_EXPORT void sg_set_device_mem(sg_device_mem_t* pmem, unsigned long long size,
                       unsigned long long addr);

/**
 * @name    bm_set_device_mem_u64
 * @brief   To fill in a device memory descriptor with size and address
 * @ingroup bmlib_runtime
 *
 * @param [in] pmem  The device memory descriptor pointer
 * @param [in]  size  The device memory descriptor's size
 * @param [in]  addr  The device memory descriptor's address
 */
DECL_EXPORT void bm_set_device_mem_u64(bm_device_mem_u64_t* pmem, unsigned long long size,
                       unsigned long long addr);

/**
 * @name    bm_mem_from_device
 * @brief   To create a device memory descriptor from address and size
 * @ingroup bmlib_runtime
 *
 * @param [in] device_addr The device memory address
 * @param [in] len         The device memory size
 * @retval bm_device_mem_t The device memory descriptor created
 */
DECL_EXPORT bm_device_mem_t bm_mem_from_device(unsigned long long device_addr,
                                   unsigned int len);

/**
 * @name    sg_mem_from_device
 * @brief   To create a device memory descriptor from address and size
 * @ingroup bmlib_runtime
 *
 * @param [in] device_addr The device memory address
 * @param [in] len         The device memory size
 * @retval bm_device_mem_t The device memory descriptor created
 */
DECL_EXPORT sg_device_mem_t sg_mem_from_device(unsigned long long device_addr,
                                   unsigned long long len);

/**
 * @name    bm_mem_from_device_u64
 * @brief   To create a device memory descriptor from address and size
 * @ingroup bmlib_runtime
 *
 * @param [in] device_addr The device memory address
 * @param [in] len         The device memory size
 * @retval bm_device_mem_t The device memory descriptor created
 */
DECL_EXPORT bm_device_mem_u64_t bm_mem_from_device_u64(unsigned long long device_addr,
                                   unsigned long long len);

/**
 * @name    bm_mem_get_system_addr
 * @brief   To get a system memory descriptor's address
 * @ingroup bmlib_runtime
 *
 * @param [in] mem  The system memory descriptor
 * @retval void *   The system memory descriptor's address
 */
DECL_EXPORT void *bm_mem_get_system_addr(struct bm_mem_desc mem);

/**
 * @name    sg_mem_get_system_addr
 * @brief   To get a system memory descriptor's address
 * @ingroup bmlib_runtime
 *
 * @param [in] mem  The system memory descriptor
 * @retval void *   The system memory descriptor's address
 */
DECL_EXPORT void *sg_mem_get_system_addr(struct sg_mem_desc mem);

/**
 * @name    bm_mem_get_system_addr_u64
 * @brief   To get a system memory descriptor's address
 * @ingroup bmlib_runtime
 *
 * @param [in] mem  The system memory descriptor
 * @retval void *   The system memory descriptor's address
 */
DECL_EXPORT void *bm_mem_get_system_addr_u64(struct bm_mem_desc_u64 mem);

/**
 * @name    bm_mem_set_system_addr
 * @brief   To set a system memory descriptor's address
 * @ingroup bmlib_runtime
 *
 * @param [in]  pmem  The system memory descriptor pointer
 * @param [in]   addr The system memory address
 */
DECL_EXPORT void bm_mem_set_system_addr(struct bm_mem_desc* pmem, void *addr);

/**
 * @name    sg_mem_set_system_addr
 * @brief   To set a system memory descriptor's address
 * @ingroup bmlib_runtime
 *
 * @param [in]  pmem  The system memory descriptor pointer
 * @param [in]   addr The system memory address
 */
DECL_EXPORT void sg_mem_set_system_addr(struct sg_mem_desc* pmem, void *addr);

/**
 * @name    bm_mem_set_system_addr_u64
 * @brief   To set a system memory descriptor's address
 * @ingroup bmlib_runtime
 *
 * @param [in]  pmem  The system memory descriptor pointer
 * @param [in]   addr The system memory address
 */
DECL_EXPORT void bm_mem_set_system_addr_u64(struct bm_mem_desc_u64* pmem, void *addr);

/**
 * @name    bm_mem_from_system
 * @brief   To create a system memory descriptor with the given system address
 * @ingroup bmlib_runtime
 *
 * @param [in]  system_addr  The system address in the descriptor
 * @retval  bm_system_mem_t  The system memory descriptor created
 */
DECL_EXPORT bm_system_mem_t bm_mem_from_system(void *system_addr);

/*******************memory alloc and free functions ***************************/
/**
 * @name    bm_mem_null
 * @brief   Return an illegal device memory descriptor
 * @ingroup bmlib_runtime
 *
 * @retval  bm_device_mem_t  An invalid device memory descriptor
 */
DECL_EXPORT bm_device_mem_t bm_mem_null(void);
#define BM_MEM_NULL (bm_mem_null())

/**
 * @name    bm_malloc_neuron_device
 * @brief   To malloc device memory according to a tensor shape
 *          (each neuron is 32 bits)
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result devcie memory descriptor
 * @param [in]  n, c, h, w  The shape of the input tensor
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_malloc_neuron_device(bm_handle_t handle, bm_device_mem_t *pmem,
                                    int n, int c, int h, int w);

/**
 * @name    sg_malloc_neuron_device
 * @brief   To malloc device memory according to a tensor shape
 *          (each neuron is 32 bits)
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result devcie memory descriptor
 * @param [in]  n, c, h, w  The shape of the input tensor
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t sg_malloc_neuron_device(bm_handle_t handle, sg_device_mem_t *pmem,
                                    unsigned long long n, unsigned long long c,
                                    unsigned long long h, unsigned long long w);

/**
 * @name    bm_malloc_neuron_device_u64
 * @brief   To malloc device memory according to a tensor shape
 *          (each neuron is 32 bits)
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result devcie memory descriptor
 * @param [in]  n, c, h, w  The shape of the input tensor
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_malloc_neuron_device_u64(bm_handle_t handle, bm_device_mem_u64_t *pmem,
                                    unsigned long long n, unsigned long long c,
                                    unsigned long long h, unsigned long long w);

/**
 * @name    bm_malloc_device_dword
 * @brief   To malloc device memory in size of dword (32 bits)
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result device memory descriptor
 * @param [in]   count  The number of dwords(32bits) to allocate
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_malloc_device_dword(bm_handle_t handle, bm_device_mem_t *pmem,
                                   int count);

/**
 * @name    sg_malloc_device_dword
 * @brief   To malloc device memory in size of dword (32 bits)
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result device memory descriptor
 * @param [in]   count  The number of dwords(32bits) to allocate
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t sg_malloc_device_dword(bm_handle_t handle, sg_device_mem_t *pmem,
                                   unsigned long long count);

/**
 * @name    bm_malloc_device_dword_u64
 * @brief   To malloc device memory in size of dword (32 bits)
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result device memory descriptor
 * @param [in]   count  The number of dwords(32bits) to allocate
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_malloc_device_dword_u64(bm_handle_t handle, bm_device_mem_u64_t *pmem,
                                   unsigned long long count);

/**
 * @name    bm_malloc_device_byte
 * @brief   To malloc device memory in size of byte
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result device memory descriptor
 * @param [in]   size   The number of bytes to allocate
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_malloc_device_byte(bm_handle_t handle, bm_device_mem_t *pmem,
                                  unsigned int size);

/**
 * @name    bm_malloc_device_mem
 * @brief   To malloc device memory in size of byte and output paddr
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  paddr  The result malloc device memory addr
 * @param [in]  heap_id The heap where to allocate  0/1/2
 * @param [in]  size    The number of bytes to allocate
 * @retval  paddr
 */
DECL_EXPORT bm_status_t bm_malloc_device_mem(bm_handle_t handle, unsigned long long *paddr,
                                              int heap_id, unsigned long long size);

/**
 * @name    bm_malloc_device_mem_mask
 * @brief   To malloc device memory in size of byte within the specified heaps and output paddr
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  paddr  The result malloc device memory addr
 * @param [in]  heap_id_mask The mask which heaps allocate from. each bit indicate one heap
 * @param [in]  size    The number of bytes to allocate
 * @retval  paddr
 */
DECL_EXPORT bm_status_t bm_malloc_device_mem_mask(bm_handle_t handle, unsigned long long *paddr,
                                              int heap_id_mask, unsigned long long size);

/**
 * @name    sg_malloc_device_byte
 * @brief   To malloc device memory in size of byte
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result device memory descriptor
 * @param [in]   size   The number of bytes to allocate
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t sg_malloc_device_byte(bm_handle_t handle, sg_device_mem_t *pmem,
                                  unsigned long long size);

/**
 * @name    bm_malloc_device_byte_u64
 * @brief   To malloc device memory in size of byte
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result device memory descriptor
 * @param [in]   size   The number of bytes to allocate
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_malloc_device_byte_u64(bm_handle_t handle, bm_device_mem_u64_t *pmem,
                                  unsigned long long size);

/**
 * @name    bm_malloc_device_byte_heap
 * @brief   To malloc device memory in size of byte within the specified heap
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result device memory descriptor
 * @param [in]  heap_id The heap where to allocate  0/1/2
 * @param [in]   size   The number of bytes to allocate
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_malloc_device_byte_heap(bm_handle_t handle, bm_device_mem_t *pmem,
                                  int heap_id, unsigned int size);

/**
 * @name    sg_malloc_device_byte_heap
 * @brief   To malloc device memory in size of byte within the specified heap
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result device memory descriptor
 * @param [in]  heap_id The heap where to allocate  0/1/2
 * @param [in]   size   The number of bytes to allocate
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t sg_malloc_device_byte_heap(bm_handle_t handle, sg_device_mem_t *pmem,
                                  int heap_id, unsigned long long size);

/**
 * @name    bm_malloc_device_byte_heap_u64
 * @brief   To malloc device memory in size of byte within the specified heap
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result device memory descriptor
 * @param [in]  heap_id The heap where to allocate  0/1/2
 * @param [in]   size   The number of bytes to allocate
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_malloc_device_byte_heap_u64(bm_handle_t handle, bm_device_mem_u64_t *pmem,
                                  int heap_id, unsigned long long size);

/**
 * @name    bm_malloc_device_byte_heap_mask
 * @brief   To malloc device memory in size of byte within the specified heaps
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result device memory descriptor
 * @param [in]  heap_id_mask The mask which heaps allocate from. each bit indicate one heap
 * @param [in]   size   The number of bytes to allocate
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_malloc_device_byte_heap_mask(bm_handle_t handle, bm_device_mem_t *pmem,
                                  int heap_id_mask, unsigned int size);

/**
 * @name    sg_malloc_device_byte_heap_mask
 * @brief   To malloc device memory in size of byte within the specified heaps
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result device memory descriptor
 * @param [in]  heap_id_mask The mask which heaps allocate from. each bit indicate one heap
 * @param [in]   size   The number of bytes to allocate
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t sg_malloc_device_byte_heap_mask(bm_handle_t handle, sg_device_mem_t *pmem,
                                  int heap_id_mask, unsigned long long size);

/**
 * @name    bm_malloc_device_byte_heap_mask_u64
 * @brief   To malloc device memory in size of byte within the specified heaps
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  pmem   The result device memory descriptor
 * @param [in]  heap_id_mask The mask which heaps allocate from. each bit indicate one heap
 * @param [in]   size   The number of bytes to allocate
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_malloc_device_byte_heap_mask_u64(bm_handle_t handle, bm_device_mem_u64_t *pmem,
                                  int heap_id_mask, unsigned long long size);

/**
 * @name    bm_free_device_mem
 * @brief   To free device memory and input paddr
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  paddr   The device memory addr to free
 */
DECL_EXPORT void bm_free_device_mem(bm_handle_t ctx, unsigned long long paddr);

/**
 * @name    bm_free_device
 * @brief   To free device memory
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  mem     The device memory descriptor to free
 */
DECL_EXPORT void bm_free_device(bm_handle_t handle, bm_device_mem_t mem);

/**
 * @name    sg_free_device
 * @brief   To free device memory
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  mem     The device memory descriptor to free
 */
DECL_EXPORT void sg_free_device(bm_handle_t handle, sg_device_mem_t mem);

/**
 * @name    bm_free_device_u64
 * @brief   To free device memory
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  mem     The device memory descriptor to free
 */
DECL_EXPORT void bm_free_device_u64(bm_handle_t handle, bm_device_mem_u64_t mem);

/**
 * @name    bm_gmem_arm_reserved_request
 * @brief   To obtain the address of global memory reserved for arm926
 * @param [in]  handle  The device handle
 *
 * @retval unsigned long long  The absolute address of gmem reserved for arm926
 */
DECL_EXPORT unsigned long long bm_gmem_arm_reserved_request(bm_handle_t handle);

/**
 * @name    bm_gmem_arm_reserved_release
 * @brief   To release the global memory reserved for arm926
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 */
DECL_EXPORT void bm_gmem_arm_reserved_release(bm_handle_t handle);

/*******************memory copy functions *************************************/
/**
 * @name    bm_memcpy_s2d
 * @brief   To copy data from system memory to device memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in] dst     The destination memory (device memory descriptor )
 * @param [in] src     The source memory (system memory, a void* pointer)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_s2d(bm_handle_t handle, bm_device_mem_t dst, void *src);

/**
 * @name    bm_memcpy_s2d_gather
 * @brief   To copy data from system virtual memory to device memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in] dst     The destination memory (device memory descriptor )
 * @param [in] argc    The number of system memory and len (system memory, a void* pointer)
 * @param [in] ...     void *src and unsigned long long len
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_s2d_gather(bm_handle_t handle, bm_device_mem_t dst, int argc, ...);

/**
 * @name    bm_memcpy_d2s_scatter
 * @brief   To copy data from  device memory to system virtual memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in] src     The destination memory (device memory descriptor )
 * @param [in] argc    The number of system memory and len (system memory, a void* pointer)
 * @param [in] ...     void *dst and unsigned long long len
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_d2s_scatter(bm_handle_t handle, bm_device_mem_t src, int argc, ...);
/**
 * @name    bm_memcpy_p2p
 * @brief   To copy data from one chip to another chip
 * @ingroup bmlib_runtime
 *
 * @param [in] handle_src The source device handle
 * @param [in] src        The source memory (device memory descriptor )
 * @param [in] handle_dst The destination device handle
 * @param [in] dst        The destination memory (device memory descriptor )
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_p2p(bm_handle_t handle_src, bm_device_mem_t src, bm_handle_t handle_dst,bm_device_mem_t dst);

/**
 * @name    sg_memcpy_s2d
 * @brief   To copy data from system memory to device memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in] dst     The destination memory (device memory descriptor )
 * @param [in] src     The source memory (system memory, a void* pointer)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t sg_memcpy_s2d(bm_handle_t handle, sg_device_mem_t dst, void *src);

/**
 * @name    bm_memcpy_s2d_u64
 * @brief   To copy data from system memory to device memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in] dst     The destination memory (device memory descriptor )
 * @param [in] src     The source memory (system memory, a void* pointer)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_s2d_u64(bm_handle_t handle, bm_device_mem_u64_t dst, void *src);

/**
 * @name    bm_memcpy_s2d_partial_offset
 * @brief   To copy specified bytes of data from system memory to device memory
 *          with an offset in device memory address.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (device memory descriptor)
 * @param [in]  src    The source memory (system memory, a void* pointer)
 * @param [in] size    The size of data to copy (in bytes)
 * @param [in] offset  The offset of the device memory address
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_s2d_partial_offset(bm_handle_t handle,
                                         bm_device_mem_t dst, void *src,
                                         unsigned int size,
                                         unsigned int offset);

/**
 * @name    sg_memcpy_s2d_partial_offset
 * @brief   To copy specified bytes of data from system memory to device memory
 *          with an offset in device memory address.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (device memory descriptor)
 * @param [in]  src    The source memory (system memory, a void* pointer)
 * @param [in] size    The size of data to copy (in bytes)
 * @param [in] offset  The offset of the device memory address
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t sg_memcpy_s2d_partial_offset(bm_handle_t handle,
                                         sg_device_mem_t dst, void *src,
                                         unsigned long long size,
                                         unsigned long long offset);

/**
 * @name    bm_memcpy_s2d_partial_offset_u64
 * @brief   To copy specified bytes of data from system memory to device memory
 *          with an offset in device memory address.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (device memory descriptor)
 * @param [in]  src    The source memory (system memory, a void* pointer)
 * @param [in] size    The size of data to copy (in bytes)
 * @param [in] offset  The offset of the device memory address
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_s2d_partial_offset_u64(bm_handle_t handle,
                                         bm_device_mem_u64_t dst, void *src,
                                         unsigned long long size,
                                         unsigned long long offset);

/**
 * @name    bm_memcpy_s2d_partial
 * @brief   To copy specified bytes of data from system memory to device memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (device memory descriptor)
 * @param [in]  src    The source memory (system memory, a void* pointer)
 * @param [in] size    The size of data to copy (in bytes)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_s2d_partial(bm_handle_t handle, bm_device_mem_t dst,
                                  void *src, unsigned int size);

/**
 * @name    sg_memcpy_s2d_partial
 * @brief   To copy specified bytes of data from system memory to device memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (device memory descriptor)
 * @param [in]  src    The source memory (system memory, a void* pointer)
 * @param [in] size    The size of data to copy (in bytes)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t sg_memcpy_s2d_partial(bm_handle_t handle, sg_device_mem_t dst,
                                  void *src, unsigned long long size);

/**
 * @name    bm_memcpy_s2d_partial_u64
 * @brief   To copy specified bytes of data from system memory to device memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (device memory descriptor)
 * @param [in]  src    The source memory (system memory, a void* pointer)
 * @param [in] size    The size of data to copy (in bytes)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_s2d_partial_u64(bm_handle_t handle, bm_device_mem_u64_t dst,
                                  void *src, unsigned long long size);

/**
 * @name    bm_memcpy_d2s
 * @brief   To copy data from device memory to system memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (system memory, a void* pointer)
 * @param [in]  src    The source memory (device memory descriptor)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_d2s(bm_handle_t handle, void *dst, bm_device_mem_t src);

/**
 * @name    sg_memcpy_d2s
 * @brief   To copy data from device memory to system memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (system memory, a void* pointer)
 * @param [in]  src    The source memory (device memory descriptor)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t sg_memcpy_d2s(bm_handle_t handle, void *dst, sg_device_mem_t src);

/**
 * @name    bm_memcpy_d2s_u64
 * @brief   To copy data from device memory to system memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (system memory, a void* pointer)
 * @param [in]  src    The source memory (device memory descriptor)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_d2s_u64(bm_handle_t handle, void *dst, bm_device_mem_u64_t src);

/**
 * @name    bm_memcpy_d2s_partial_offset
 * @brief   To copy specified bytes of data from device memory to system memory
 *          with an offset in device memory address.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (system memory, a void* pointer)
 * @param [in]  src    The source memory (device memory descriptor)
 * @param [in] size    The size of data to copy (in bytes)
 * @param [in] offset  The offset of the device memory address
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_d2s_partial_offset(bm_handle_t handle, void *dst,
                                         bm_device_mem_t src, unsigned int size,
                                         unsigned int offset);

/**
 * @name    sg_memcpy_d2s_partial_offset
 * @brief   To copy specified bytes of data from device memory to system memory
 *          with an offset in device memory address.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (system memory, a void* pointer)
 * @param [in]  src    The source memory (device memory descriptor)
 * @param [in] size    The size of data to copy (in bytes)
 * @param [in] offset  The offset of the device memory address
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t sg_memcpy_d2s_partial_offset(bm_handle_t handle, void *dst,
                                         sg_device_mem_t src, unsigned long long size,
                                         unsigned long long offset);

/**
 * @name    bm_memcpy_d2s_partial_offset_u64
 * @brief   To copy specified bytes of data from device memory to system memory
 *          with an offset in device memory address.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (system memory, a void* pointer)
 * @param [in]  src    The source memory (device memory descriptor)
 * @param [in] size    The size of data to copy (in bytes)
 * @param [in] offset  The offset of the device memory address
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_d2s_partial_offset_u64(bm_handle_t handle, void *dst,
                                         bm_device_mem_u64_t src, unsigned long long size,
                                         unsigned long long offset);

/**
 * @name    bm_memcpy_d2s_partial
 * @brief   To copy specified bytes of data from device memory to system memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (system memory, a void* pointer)
 * @param [in]  src    The source memory (device memory descriptor)
 * @param [in] size    The size of data to copy (in bytes)
 *
 * @retval  BM_SUCCESS  Data transfer succeeds.
 *          Other code  Data transfer fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_d2s_partial(bm_handle_t handle, void *dst,
                                  bm_device_mem_t src, unsigned int size);

/**
 * @name    sg_memcpy_d2s_partial
 * @brief   To copy specified bytes of data from device memory to system memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (system memory, a void* pointer)
 * @param [in]  src    The source memory (device memory descriptor)
 * @param [in] size    The size of data to copy (in bytes)
 *
 * @retval  BM_SUCCESS  Data transfer succeeds.
 *          Other code  Data transfer fails.
 */
DECL_EXPORT bm_status_t sg_memcpy_d2s_partial(bm_handle_t handle, void *dst,
                                  sg_device_mem_t src, unsigned long long size);

/**
 * @name    bm_memcpy_d2s_partial_u64
 * @brief   To copy specified bytes of data from device memory to system memory
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in]  dst    The destination memory (system memory, a void* pointer)
 * @param [in]  src    The source memory (device memory descriptor)
 * @param [in] size    The size of data to copy (in bytes)
 *
 * @retval  BM_SUCCESS  Data transfer succeeds.
 *          Other code  Data transfer fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_d2s_partial_u64(bm_handle_t handle, void *dst,
                                  bm_device_mem_u64_t src, unsigned long long size);

/**
 * @name    bm_memcpy_d2d
 * @brief   To copy specified dwords of data from one piece of device memory
 *          to another piece of device memory within one device. Both source
 *          and destination offsets can be specified.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle     The device handle
 * @param [in]  dst       The destination device memory
 * @param [in] dst_offset The offset of destination device memory address
 * @param [in]  src       The source device memory
 * @param [in] src_offset The offset of source device memory address
 * @param [in]  len       Length of data to copy (in DWORD 4 bytes)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_d2d(bm_handle_t handle, bm_device_mem_t dst,
                          int dst_offset, bm_device_mem_t src, int src_offset,
                          int len);

/**
 * @name    bm_memcpy_d2d_with_core
 * @brief   To copy specified dwords of data from one piece of device memory
 *          to another piece of device memory within one device. Both source
 *          and destination offsets can be specified.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle     The device handle
 * @param [in]  dst       The destination device memory
 * @param [in] dst_offset The offset of destination device memory address
 * @param [in]  src       The source device memory
 * @param [in] src_offset The offset of source device memory address
 * @param [in]  len       Length of data to copy (in DWORD 4 bytes)
 * @param [in] core_id    The core id to copy
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_d2d_with_core(bm_handle_t handle, bm_device_mem_t dst,
                          int dst_offset, bm_device_mem_t src, int src_offset,
                          int len, int core_id);

/**
 * @name    bm_memcpy_d2d_byte
 * @brief   To copy specified bytes of data from one piece of device memory
 *          to another piece of device memory within one device. Both source
 *          and destination offsets can be specified.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle     The device handle
 * @param [in]  dst       The destination device memory
 * @param [in] dst_offset The offset of destination device memory address (in bytes)
 * @param [in]  src       The source device memory
 * @param [in] src_offset The offset of source device memory address (in bytes)
 * @param [in]  size      Size of data to copy (in bytes)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_d2d_byte(bm_handle_t handle, bm_device_mem_t dst,
                               size_t dst_offset, bm_device_mem_t src,
                               size_t src_offset, size_t size);

/**
 * @name    bm_memcpy_d2d_byte_with_core
 * @brief   To copy specified bytes of data from one piece of device memory
 *          to another piece of device memory within one device. Both source
 *          and destination offsets can be specified.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle     The device handle
 * @param [in]  dst       The destination device memory
 * @param [in] dst_offset The offset of destination device memory address (in bytes)
 * @param [in]  src       The source device memory
 * @param [in] src_offset The offset of source device memory address (in bytes)
 * @param [in]  size      Size of data to copy (in bytes)
 * @param [in] core_id    The core id to copy
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_d2d_byte_with_core(bm_handle_t handle, bm_device_mem_t dst,
                               size_t dst_offset, bm_device_mem_t src,
                               size_t src_offset, size_t size, int core_id);

/**
 * @name    bm_memcpy_d2d_stride
 * @brief   To copy specified data from one piece of device memory
 *          to another piece of device memory within one device. Both source
 *          and destination offsets can be specified.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle      The device handle
 * @param [in] dst         The destination device memory
 * @param [in] dst_stride  The data stride of destination data
 * @param [in] src         The source device memory
 * @param [in] src_stride  The data stride of source data
 * @param [in] count       Count of data to copy
 * @param [in] format_size Data format byte size, such as sizeof(uint8_t), sizeof(float), etc.
 *                         format_size only support 1/2/4.
 *
 * dst_stride MUST be 1, EXCEPT: dst_stride == 4 && src_stride == 1 && format_size ==1
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_d2d_stride(bm_handle_t     handle,
                                 bm_device_mem_t dst,
                                 int             dst_stride,
                                 bm_device_mem_t src,
                                 int             src_stride,
                                 int             count,
                                 int             format_size);

/**
 * @name    bm_memcpy_d2d_stride
 * @brief   To copy specified data from one piece of device memory
 *          to another piece of device memory within one device. Both source
 *          and destination offsets can be specified.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle      The device handle
 * @param [in] dst         The destination device memory
 * @param [in] dst_stride  The data stride of destination data
 * @param [in] src         The source device memory
 * @param [in] src_stride  The data stride of source data
 * @param [in] count       Count of data to copy
 * @param [in] format_size Data format byte size, such as sizeof(uint8_t), sizeof(float), etc.
 *                         format_size only support 1/2/4.
 * @param [in] core_id     The core id to copy.
 *
 * dst_stride MUST be 1, EXCEPT: dst_stride == 4 && src_stride == 1 && format_size ==1
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_d2d_stride_with_core(bm_handle_t     handle,
                                 bm_device_mem_t dst,
                                 int             dst_stride,
                                 bm_device_mem_t src,
                                 int             src_stride,
                                 int             count,
                                 int             format_size,
                                 int             core_id);

/**
 * @name    bm_memcpy_d2d_u64
 * @brief   To copy specified dwords of data from one piece of device memory
 *          to another piece of device memory within one device. Both source
 *          and destination offsets can be specified.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle     The device handle
 * @param [in] dst        The destination device memory
 * @param [in] dst_offset The offset of destination device memory address
 * @param [in] src        The source device memory
 * @param [in] src_offset The offset of source device memory address
 * @param [in] len        Length of data to copy (in DWORD 4 bytes)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_d2d_u64(bm_handle_t handle,
                                 bm_device_mem_u64_t  dst,
                                 unsigned long long   dst_offset,
                                 bm_device_mem_u64_t  src,
                                 unsigned long long   src_offset,
                                 unsigned long long   len);

/**
 * @name    bm_memcpy_d2d_byte_u64
 * @brief   To copy specified bytes of data from one piece of device memory
 *          to another piece of device memory within one device. Both source
 *          and destination offsets can be specified.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle     The device handle
 * @param [in] dst        The destination device memory
 * @param [in] dst_offset The offset of destination device memory address (in bytes)
 * @param [in] src        The source device memory
 * @param [in] src_offset The offset of source device memory address (in bytes)
 * @param [in] size       Size of data to copy (in bytes)
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_d2d_byte_u64(bm_handle_t handle,
                                    bm_device_mem_u64_t dst,
                                    unsigned long long  dst_offset,
                                    bm_device_mem_u64_t src,
                                    unsigned long long  src_offset,
                                    unsigned long long  size);

/**
 * @name    bm_memcpy_d2d_stride_u64
 * @brief   To copy specified data from one piece of device memory
 *          to another piece of device memory within one device. Both source
 *          and destination offsets can be specified.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle      The device handle
 * @param [in] dst         The destination device memory
 * @param [in] dst_stride  The data stride of destination data
 * @param [in] src         The source device memory
 * @param [in] src_stride  The data stride of source data
 * @param [in] count       Count of data to copy
 * @param [in] format_size Data format byte size, such as sizeof(uint8_t), sizeof(float), etc.
 *                         format_size only support 1/2/4.
 *
 * dst_stride MUST be 1, EXCEPT: dst_stride == 4 && src_stride == 1 && format_size ==1
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_memcpy_d2d_stride_u64(bm_handle_t          handle,
                                      bm_device_mem_u64_t dst,
                                      unsigned long long  dst_stride,
                                      bm_device_mem_u64_t src,
                                      unsigned long long  src_stride,
                                      unsigned long long  count,
                                      int                 format_size);

/**
 * @name    bm_memcpy_c2c
 * @brief   To copy data from one chip to another chip.
 *          (Used in multi-chip card scenario)
 * @ingroup bmlib_runtime
 *
 * @param [in] src_handle The source device handle
 * @param [in] dst_handle The destination device handle
 * @param [in] src        The source device memory descriptor
 * @param [in] dst        The destination device memory descriptor
 * @param [in] force_dst_cdma If use the CDMA engine of the destination device
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memcpy_c2c(bm_handle_t src_handle, bm_handle_t dst_handle,
                          bm_device_mem_t src, bm_device_mem_t dst,
                          bool force_dst_cdma);

/**
 * @name    bm_memset_device
 * @brief   To fill in specified device memory with the given value
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   value  The value used to fill. (int type)
 * @param [in]  mem     The device memory which will be filled in
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memset_device(bm_handle_t handle, const int value,
                             bm_device_mem_t mem);

/**
 * @name    bm_memset_device_ext
 * @brief   To fill in specified device memory with the given value and mode
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   value  The pointer of value used to fill
 * @param [in]   mode   The valid bytes of *value
 * @param [in]  mem     The device memory which will be filled in
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memset_device_ext(bm_handle_t handle, void* value, int mode,
                             bm_device_mem_t mem);

/**
 * @name    bm_memset_device_ext_u64
 * @brief   To fill in specified device memory with the given value and mode
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  value   The pointer of value used to fill
 * @param [in]  mode    The valid bytes of *value
 * @param [in]  mem     The device memory which will be filled in
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_memset_device_ext_u64(bm_handle_t handle, void* value, int mode,
                             bm_device_mem_u64_t mem);

/**
 * @name    bm_mem_convert_system_to_device_neuron
 * @brief   To malloc a piece of device memory according to the shape of
 *          neuron(in DWORD 4 bytes); copy neuron from system memory to
 *          device memory if need_copy is true.
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  dev_mem The device memory descriptor
 * @param [in]  sys_mem The system memory descriptor
 * @param [in]  need_copy If copy from system to device is needed
 * @param [in]  n,c,h,w  Neuron shape size
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_convert_system_to_device_neuron(bm_handle_t handle,
                                                   struct bm_mem_desc *dev_mem,
                                                   struct bm_mem_desc sys_mem,
                                                   bool need_copy, int n, int c,
                                                   int h, int w);

/**
 * @name    bm_mem_convert_system_to_device_neuron_byte
 * @brief   To malloc a piece of device memory according to the shape of
 *          neuron(in bytes); copy neuron from system memory to
 *          device memory if need_copy is true.
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  dev_mem The device memory descriptor
 * @param [in]  sys_mem The system memory descriptor
 * @param [in]  need_copy If copy from system to device is needed
 * @param [in]  n,c,h,w  Neuron shape size
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_convert_system_to_device_neuron_byte(
    bm_handle_t handle, struct bm_mem_desc *dev_mem, struct bm_mem_desc sys_mem,
    bool need_copy, int n, int c, int h, int w);

/**
 * @name    bm_mem_convert_system_to_device_coeff
 * @brief   To malloc a piece of device memory according to the size of
 *          coefficient (in DWORD 4 bytes); copy coefficient from system
 *          memory to device memory if need_copy is true.
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  dev_mem The device memory descriptor
 * @param [in]  sys_mem The system memory descriptor
 * @param [in]  need_copy If copy from system to device is needed
 * @param [in]  coeff_count Coefficient size
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_convert_system_to_device_coeff(bm_handle_t handle,
                                                  struct bm_mem_desc *dev_mem,
                                                  struct bm_mem_desc sys_mem,
                                                  bool need_copy,
                                                  int coeff_count);
/**
 * @name    bm_mem_convert_system_to_device_coeff_byte
 * @brief   To malloc a piece of device memory according to the size of
 *          coefficient (in bytes); copy coefficient from system
 *          memory to device memory if need_copy is true.
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  dev_mem The device memory descriptor
 * @param [in]  sys_mem The system memory descriptor
 * @param [in]  need_copy If copy from system to device is needed
 * @param [in]  coeff_count Coefficient size
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_convert_system_to_device_coeff_byte(
    bm_handle_t handle, struct bm_mem_desc *dev_mem, struct bm_mem_desc sys_mem,
    bool need_copy, int coeff_count);

/*******************memory map functions *************************************/
/**
 * @name    bm_mem_mmap_device_mem
 * @brief   To map a piece of device memory to user space with cache enabled.
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  dev_mem The device memory to map
 * @param [out] vmem    The virtual address of the mapped device memory
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_mmap_device_mem(bm_handle_t handle, bm_device_mem_t *dmem,

        unsigned long long *vmem);

/**
 * @name    sg_mem_mmap_device_mem
 * @brief   To map a piece of device memory to user space with cache enabled.
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  dev_mem The device memory to map
 * @param [out] vmem    The virtual address of the mapped device memory
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t sg_mem_mmap_device_mem(bm_handle_t handle, sg_device_mem_t *dmem,
        unsigned long long *vmem);

/**
 * @name    bm_mem_mmap_device_mem_u64
 * @brief   To map a piece of device memory to user space with cache enabled.
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  dev_mem The device memory to map
 * @param [out] vmem    The virtual address of the mapped device memory
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_mmap_device_mem_u64(bm_handle_t handle, bm_device_mem_u64_t *dmem,
        unsigned long long *vmem);

/*******************memory map functions *************************************/
/**
 * @name    bm_mem_mmap_device_mem_no_cache
 * @brief   To map a piece of device memory to user space with cache disabled.
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  dev_mem The device memory to map
 * @param [out] vmem    The virtual address of the mapped device memory
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_mmap_device_mem_no_cache(bm_handle_t handle, bm_device_mem_t *dmem,

        unsigned long long *vmem);

/**
 * @name    sg_mem_mmap_device_mem_no_cache
 * @brief   To map a piece of device memory to user space with cache disabled.
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  dev_mem The device memory to map
 * @param [out] vmem    The virtual address of the mapped device memory
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t sg_mem_mmap_device_mem_no_cache(bm_handle_t handle, sg_device_mem_t *dmem,
        unsigned long long *vmem);

/**
 * @name    bm_mem_mmap_device_mem_no_cache_u64
 * @brief   To map a piece of device memory to user space with cache disabled.
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  dev_mem The device memory to map
 * @param [out] vmem    The virtual address of the mapped device memory
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_mmap_device_mem_no_cache_u64(bm_handle_t handle, bm_device_mem_u64_t *dmem,
        unsigned long long *vmem);

/**
 * @name    bm_mem_vir_to_phy
 * @brief   To get device mem address through the mapped virtual address .
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  vmem    The virtual address of the mapped device memory
 * @param [out]  dev_mem The device memory address
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_vir_to_phy(bm_handle_t handle, unsigned long long vmem,
        unsigned long long *device_mem);
/**
 * @name    bm_mem_invalidate_device_mem
 * @brief   To invalidate a piece of mapped device memory to maintain
 *          cache coherence
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   dmem   The device memory to invalidate
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */

DECL_EXPORT bm_status_t bm_mem_invalidate_device_mem(bm_handle_t handle,
                                         bm_device_mem_t *dmem);

/**
 * @name    sg_mem_invalidate_device_mem
 * @brief   To invalidate a piece of mapped device memory to maintain
 *          cache coherence
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   dmem   The device memory to invalidate
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */

DECL_EXPORT bm_status_t sg_mem_invalidate_device_mem(bm_handle_t handle,
                                         sg_device_mem_t *dmem);

/**
 * @name    bm_mem_invalidate_device_mem_u64
 * @brief   To invalidate a piece of mapped device memory to maintain
 *          cache coherence
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   dmem   The device memory to invalidate
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */

DECL_EXPORT bm_status_t bm_mem_invalidate_device_mem_u64(bm_handle_t handle,
                                         bm_device_mem_u64_t *dmem);

/**
 * @name    bm_mem_invalidate_partial_device_mem
 * @brief   To invalidate part of mapped device memory to maintain
 *          cache coherence
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   dmem   The device memory to invalidate
 * @param [in]  offset  The offset of device memory address
 * @param [in]  len     The length of memory to invalidate in bytes
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_invalidate_partial_device_mem(bm_handle_t handle,
                                                 bm_device_mem_t *dmem,
                                                 unsigned int offset,
                                                 unsigned int len);

/**
 * @name    sg_mem_invalidate_partial_device_mem
 * @brief   To invalidate part of mapped device memory to maintain
 *          cache coherence
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   dmem   The device memory to invalidate
 * @param [in]  offset  The offset of device memory address
 * @param [in]  len     The length of memory to invalidate in bytes
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t sg_mem_invalidate_partial_device_mem(bm_handle_t handle,
                                                 sg_device_mem_t *dmem,
                                                 unsigned long long offset,
                                                 unsigned long long len);

/**
 * @name    bm_mem_invalidate_partial_device_mem_u64
 * @brief   To invalidate part of mapped device memory to maintain
 *          cache coherence
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   dmem   The device memory to invalidate
 * @param [in]  offset  The offset of device memory address
 * @param [in]  len     The length of memory to invalidate in bytes
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_invalidate_partial_device_mem_u64(bm_handle_t handle,
                                                 bm_device_mem_u64_t *dmem,
                                                 unsigned long long offset,
                                                 unsigned long long len);

/**
 * @name    bm_mem_flush_device_mem
 * @brief   To flush a piece of mapped device memory to maintain
 *          cache coherence
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   dmem   The device memory to flush
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_flush_device_mem(bm_handle_t handle, bm_device_mem_t *dmem);

/**
 * @name    sg_mem_flush_device_mem
 * @brief   To flush a piece of mapped device memory to maintain
 *          cache coherence
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   dmem   The device memory to flush
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t sg_mem_flush_device_mem(bm_handle_t handle, sg_device_mem_t *dmem);

/**
 * @name    bm_mem_flush_device_mem_u64
 * @brief   To flush a piece of mapped device memory to maintain
 *          cache coherence
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   dmem   The device memory to flush
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_flush_device_mem_u64(bm_handle_t handle, bm_device_mem_u64_t *dmem);

/**
 * @name    bm_mem_flush_partial_device_mem
 * @brief   To flush part of mapped device memory to maintain
 *          cache coherence
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   dmem   The device memory to flush
 * @param [in]  offset  The offset of device memory address
 * @param [in]  len     The length of memory to flush in bytes
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_flush_partial_device_mem(bm_handle_t handle,
                                            bm_device_mem_t *dmem,
                                            unsigned int offset,
                                            unsigned int len);

/**
 * @name    sg_mem_flush_partial_device_mem
 * @brief   To flush part of mapped device memory to maintain
 *          cache coherence
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   dmem   The device memory to flush
 * @param [in]  offset  The offset of device memory address
 * @param [in]  len     The length of memory to flush in bytes
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t sg_mem_flush_partial_device_mem(bm_handle_t handle,
                                            sg_device_mem_t *dmem,
                                            unsigned long long offset,
                                            unsigned long long len);

/**
 * @name    bm_mem_flush_partial_device_mem_u64
 * @brief   To flush part of mapped device memory to maintain
 *          cache coherence
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   dmem   The device memory to flush
 * @param [in]  offset  The offset of device memory address
 * @param [in]  len     The length of memory to flush in bytes
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_flush_partial_device_mem_u64(bm_handle_t handle,
                                            bm_device_mem_u64_t *dmem,
                                            unsigned long long offset,
                                            unsigned long long len);

/**
 * @name    bm_mem_unmap_device_mem
 * @brief   To unmap a piece of mapped device memory
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   vmem   The virtual address of the mapped device memory
 * @param [in]  size    The size of unmapped memory
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_unmap_device_mem(bm_handle_t handle, void *vmem, int size);

/**
 * @name    sg_mem_unmap_device_mem
 * @brief   To unmap a piece of mapped device memory
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   vmem   The virtual address of the mapped device memory
 * @param [in]  size    The size of unmapped memory
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t sg_mem_unmap_device_mem(bm_handle_t handle, void *vmem, unsigned long long size);

/**
 * @name    bm_mem_unmap_device_mem_u64
 * @brief   To unmap a piece of mapped device memory
 *          (only valid in SoC mode; Not supported in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   vmem   The virtual address of the mapped device memory
 * @param [in]  size    The size of unmapped memory
 *
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_mem_unmap_device_mem_u64(bm_handle_t handle, void *vmem, unsigned long long size);

/*******************api(kernel) functions *************************************/
/**
 * @name    bm_flush
 * @brief   To synchronize APIs of the current thread. The thread will block
 *          until all the outstanding APIs of the current thread are finished.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 */
DECL_EXPORT void bm_flush(bm_handle_t handle);

/**
 * @name    bm_device_sync
 * @brief   To synchronize APIs of the device. The thread will block
 *          until all the outstanding APIs of the device are finished.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle   The device handle
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_device_sync(bm_handle_t handle);

/**
 * @name    bm_handle_sync
 * @brief   To synchronize APIs of the handle. The thread will block
 *          until all the outstanding APIs of the handle are finished.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle   The device handle
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_handle_sync(bm_handle_t handle);

/**
 * @name    bm_handle_sync_from_core
 * @brief   To synchronize APIs of the handle. The thread will block
 *          until all the outstanding APIs of the handle are finished.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle   The device handle
 * @param [in] core_id  The core id
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_handle_sync_from_core(bm_handle_t handle, int core_id);

/**
 * @name    bm_thread_sync
 * @brief   To synchronize APIs of the current thread. The thread will block
 *          until all the outstanding APIs of the current thread are finished.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @retval  BM_SUCCESS Succeeds.
 *          Other code Fails.
 */
DECL_EXPORT bm_status_t bm_thread_sync(bm_handle_t handle);

/**
 * @name    bm_set_sync_timeout
 * @brief   To set sync timeout ms.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in] timeout Sync timeout
 * @retval  BM_SUCCESS Succeeds.
 *          Other code Fails.
 */
DECL_EXPORT bm_status_t bm_set_sync_timeout(bm_handle_t handle, int timeout);

/**
 * @name    bm_thread_sync_from_core
 * @brief   To synchronize APIs of the current thread. The thread will block
 *          until all the outstanding APIs of the current thread are finished.
 * @ingroup bmlib_runtime
 *
 * @param [in] handle  The device handle
 * @param [in] core_id The core id
 * @retval  BM_SUCCESS Succeeds.
 *          Other code Fails.
 */
DECL_EXPORT bm_status_t bm_thread_sync_from_core(bm_handle_t handle, int core_id);

/*******************trace and profile releated functions **********************/
typedef struct bm_profile {
#ifdef __linux__
  unsigned long cdma_in_time;
  unsigned long cdma_in_counter;
  unsigned long cdma_out_time;
  unsigned long cdma_out_counter;
  unsigned long tpu_process_time;
  unsigned long tpu1_process_time;
  unsigned long sent_api_counter;
  unsigned long completed_api_counter;
#else
  unsigned long long cdma_in_time;
  unsigned long long cdma_in_counter;
  unsigned long long cdma_out_time;
  unsigned long long cdma_out_counter;
  unsigned long long tpu_process_time;
  unsigned long long tpu1_process_time;
  unsigned long long sent_api_counter;
  unsigned long long completed_api_counter;
#endif
} bm_profile_t;
/**
 * @name    bm_get_profile
 * @brief   To get the profile data at the moment
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out] profile The result profile data
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_profile(bm_handle_t handle, bm_profile_t *profile);

typedef struct bootloader_version{
	char *bl1_version;
	char *bl2_version;
	char *bl31_version;
	char *uboot_version;
} boot_loader_version;

/**
 * @name    bm_get_boot_loader_version
 * @brief   To get the boot_loader_version
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out] version The result version data
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_boot_loader_version(bm_handle_t handle, boot_loader_version *version);

/**
 * @name    bm_get_vpu_instant_usage
 * @brief   To get vpu usage
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out] smi_attr The result vpu usage
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_vpu_instant_usage(bm_handle_t handle, int *vpu_usage);

/**
 * @name    bm_get_jpu_core_usage
 * @brief   To get the jpu usage
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out] smi_attr The result jpu usage
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_jpu_core_usage(bm_handle_t handle, int *jpu_usage);

/**
 * @name    bm_get_vpp_instant_usage
 * @brief   To get the vpp usage
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out] smi_attr The result vpp usage
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_vpp_instant_usage(bm_handle_t handle, int *vpp_usage);
/**
 * @name    bm_get_last_api_process_time_us
 * @brief   This function is abandoned.
 */
#ifdef __linux__
DECL_EXPORT bm_status_t bm_get_last_api_process_time_us(bm_handle_t handle,
                                            unsigned long *time_us);
#else
DECL_EXPORT bm_status_t bm_get_last_api_process_time_us(bm_handle_t handle,
											unsigned long long *time_us);
#endif
/*******************tpu clock and module reset releated functions *************/

/**
 * @name    bm_set_clk_tpu_freq
 * @brief   To set the clock frequency of TPU (only valid in PCIE mode).
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]   freq   The TPU target frequency
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_set_clk_tpu_freq(bm_handle_t handle, int freq);

/**
 * @name    bm_get_clk_tpu_freq
 * @brief   To get the clock frequency of TPU
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out]  freq   The current TPU frequency
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_clk_tpu_freq(bm_handle_t handle, int *freq);

/*******************misc functions ********************************************/
struct bm_misc_info {
  int pcie_soc_mode;  /*0---pcie; 1---soc*/
  int ddr_ecc_enable; /*0---disable; 1---enable*/
  long long ddr0a_size;
  long long ddr0b_size;
  long long ddr1_size;
  long long ddr2_size;
  unsigned int chipid;
#define BM1682_CHIPID_BIT_MASK (0X1 << 0)
#define BM1684_CHIPID_BIT_MASK (0X1 << 1)
#define BM1686_CHIPID_BIT_MASK (0X1 << 2)
#ifdef __linux__
  unsigned long chipid_bit_mask;
#else
	unsigned long long chipid_bit_mask;
#endif
  unsigned int driver_version;
  int domain_bdf;
  int board_version; /*hardware board version [23:16]-mcu sw version, [15:8]-board type, [7:0]-hw version*/
  int a53_enable;
  int dyn_enable;
};

/**
 * @name    bm_get_misc_info
 * @brief   To get miscellaneous information of the device
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle     The device handle
 * @param [out] pmisc_info The fetched misc info
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_misc_info(bm_handle_t handle, struct bm_misc_info *pmisc_info);

/**
 * @name    bm_get_chipid
 * @brief   To get the chipid of the device. (0x1682 / 0x1684 / 0x168?)
 * @ingroup bmlib_runtime
 *
 * @param [in] handle    The device handle
 * @param [out] p_chipid The chip id of the device
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_chipid(bm_handle_t handle, unsigned int *p_chipid);

#define BMLIB_LOG_QUIET    -8
#define BMLIB_LOG_PANIC     0
#define BMLIB_LOG_FATAL     8
#define BMLIB_LOG_ERROR    16
#define BMLIB_LOG_WARNING  24
#define BMLIB_LOG_INFO     32
#define BMLIB_LOG_VERBOSE  40
#define BMLIB_LOG_DEBUG    48
#define BMLIB_LOG_TRACE    56

/**
 * @name    bmlib_log_get_level
 * @brief   To get the bmlib log level
 * @ingroup bmlib_log
 *
 * @param void
 * @retval  The level of bmlib log level
 */
DECL_EXPORT int  bmlib_log_get_level(void);

/**
 * @name    bmlib_log_set_level
 * @brief   To set the bmlib log level
 * @ingroup bmlib_log
 *
 * @param [in] level    The level of bmlib log level
 * @retval  void
 */
DECL_EXPORT void bmlib_log_set_level(int level);

/**
 * @name    bmlib_log_set_callback
 * @brief   To set callback to get bmlib log
 * @ingroup bmlib_log
 *
 * @param [in]  callback     The callback function to get bmlib log
 * @retval  void
 */
DECL_EXPORT void bmlib_log_set_callback(void (*callback)(const char*, int, const char*, va_list args));

/**
 * @name    bm_set_debug_mode
 * @brief   To set the debug mode for firmware log for tpu
 * @ingroup bmlib_log
 *
 * @param [in]  handle  The device handle
 * @param [in]  mode    The debug mode of fw log, 0/1 for disable/enable log
 * @retval  void
 */
DECL_EXPORT void bm_set_debug_mode(bm_handle_t handle, int mode);

/**
 * @name    bmlib_api_dbg_callback
 * @brief   To set debug callback to get firmware log
 * @ingroup bmlib_log
 *
 * @param [in]  bmlib_api_dbg_callback  callback to get firmware log
 * @retval  void
 */
typedef void (*bmlib_api_dbg_callback)(int, int, int, const char*);
// api, result, duratioin, log, third int for api duration for future
DECL_EXPORT void bmlib_set_api_dbg_callback(bmlib_api_dbg_callback callback);

/**
 * @name    bmcpu_get_cpu_status
 * @brief   Get bmcpu status
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @retval  BMCPU_RUNNING  bmcpu is running.
 *          Other code  Fails.
 */
DECL_EXPORT bm_cpu_status_t bmcpu_get_cpu_status(bm_handle_t handle);

/**
 * @name    bmcpu_start_cpu
 * @brief   Start cpu in pcie mode
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  boot_file       Fip file
 * @param [in]  core_file       Itb file
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bmcpu_start_cpu(bm_handle_t handle, char *boot_file, char *core_file);

/**
 * @name    bmcpu_open_process
 * @brief   Open a process to do some work
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  flags           Process flags
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  >= 0 process handle
 *          < 0  Other code Fails.
 */
DECL_EXPORT int bmcpu_open_process(bm_handle_t handle, unsigned int flags, int timeout);

/**
 * @name    bmcpu_load_library
 * @brief   Load a share library(so) to specific process
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  process_handle  Process handle
 * @param [in]  library_file    Library file path
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bmcpu_load_library(bm_handle_t handle, int process_handle, char *library_file, int timeout);

/**
 * @name    bmcpu_unload_library
 * @brief   Load a share library(so) to specific process
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  process_handle  Process handle
 * @param [in]  library_file    Library file path
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bmcpu_unload_library(bm_handle_t handle, int process_handle, char *library_file, int timeout);

/**
 * @name    bmcpu_exec_function
 * @brief   Execute specific function in specific process
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  process_handle  Process handle
 * @param [in]  function_name   Function name
 * @param [in]  function_param  Function parameters
 * @param [in]  param_size      Parameters size in bytes
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  0   success.
 *          >0  code fails from bmlib
 *          <0  code fails from function
 */
DECL_EXPORT int bmcpu_exec_function(bm_handle_t handle,
                     int process_handle,
                     char *function_name,
                     void *function_param,
                     unsigned int param_size,
                     int timeout);

#define BMCPU_EXEC_OPT_NO_FLUSH_CACHE     1
/**
 * @name    bmcpu_exec_function_ext
 * @brief   Execute specific function in specific process
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  process_handle  Process handle
 * @param [in]  function_name   Function name
 * @param [in]  function_param  Function parameters
 * @param [in]  param_size      Parameters size in bytes
 * @param [in]  opt             exec options
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  0   success.
 *          >0  code fails from bmlib
 *          <0  code fails from function
 */
DECL_EXPORT int bmcpu_exec_function_ext(bm_handle_t  handle,
                            int process_handle,
                            char *function_name,
                            void *function_param,
                            unsigned int param_size,
                            unsigned int opt,
                            int timeout);

/**
 * @name    bmcpu_exec_function_async
 * @brief   Execute specific function in specific process asynchronous
 *          user should use bm_query_exec_function_result to query result
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  process_handle  Process handle
 * @param [in]  function_name   Function name
 * @param [in]  function_param  Function param
 * @param [in]  param_size      Param size in bytes
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bmcpu_exec_function_async(bm_handle_t handle,
                                   int process_handle,
                                   char *function_name,
                                   void *function_param,
                                   unsigned int param_size,
                                   unsigned long long *api_handle);

/**
 * @name    bmcpu_exec_function_async_ext
 * @brief   Execute specific function in specific process asynchronous
 *          user should use bm_query_exec_function_result to query result
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  process_handle  Process handle
 * @param [in]  function_name   Function name
 * @param [in]  function_param  Function param
 * @param [in]  param_size      Param size in bytes
 * @param [in]  opt             exec options
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bmcpu_exec_function_async_ext(bm_handle_t handle,
                                          int process_handle,
                                          char *function_name,
                                          void *function_param,
                                          unsigned int param_size,
                                          unsigned int opt,
                                          unsigned long long *api_handle);

/**
 * @name    bmcpu_query_exec_function_result
 * @brief   Query result from function called by bm_exec_function
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  api_handle      Api handle return by bm_exec_function_async
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  0   success.
 *          >0  code fails from bmlib
 *          <0  code fails from function
 */
DECL_EXPORT int bmcpu_query_exec_function_result(bm_handle_t handle, unsigned long long api_handle, int timeout);

/**
 * @name    bmcpu_map_phys_addr
 * @brief   Map physical address in specific process
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  process_handle  Process handle
 * @param [in]  phys_addr       Physical address
 * @param [in]  size            Map size in bytes
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  >0  virtual address
 *          0   fails
 */
DECL_EXPORT void *bmcpu_map_phys_addr(bm_handle_t handle, int process_handle, void *phys_addr, unsigned int size, int timeout);

/**
 * @name    bmcpu_unmap_phys_addr
 * @brief   Unmap physical address in specific process
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  process_handle  Process handle
 * @param [in]  phys_addr       Physical address
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  <0  fail
 *          0   success
 */
DECL_EXPORT bm_status_t bmcpu_unmap_phys_addr(bm_handle_t handle, int process_handle, void *phys_addr, int timeout);

/**
 * @name    bmcpu_close_process
 * @brief   Close process
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  process_handle  Process handle
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bmcpu_close_process(bm_handle_t handle, int process_handle, int timeout);

/**
 * @name    bmcpu_reset_cpu
 * @brief   Reset cpu in pcie mode
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bmcpu_reset_cpu(bm_handle_t handle);

/**
 * @name    bm_enable_perf_monitor
 * @brief   enable perf monitor to get gdma and tpu performance data
 * @ingroup bmlib_perf
 *
 * @param [in]  handle         The device handle
 * @param [in]  perf_monitor   The monitor to perf
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_enable_perf_monitor(bm_handle_t handle, bm_perf_monitor_t *perf_monitor);

/**
 * @name    bm_disable_perf_monitor
 * @brief   disable perf monitor to get gdma and tpu performance data
 * @ingroup bmlib_perf
 *
 * @param [in]  handle         The device handle
 * @param [in]  perf_monitor   The monitor to perf
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_disable_perf_monitor(bm_handle_t handle, bm_perf_monitor_t *perf_monitor);

/**
 * @name    bmcpu_set_log
 * @brief   Set cpu log options
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  log_level       0: DEBUG  1:INFO 2:WARN 3:ERROR 4:FATAL
 * @param [in]  log_to_console  1: YES  0: No
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bmcpu_set_log(bm_handle_t handle, unsigned int log_level,  unsigned int log_to_console, int timeout);

/**
 * @name    bmcpu_get_log
 * @brief   Get cpu log file
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @param [in]  process_handle  Process handle
 * @param [in]  log_file        save log as file
 * @param [in]  timeout         Timeout value in millisecond, -1 means default value of this device
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bmcpu_get_log(bm_handle_t handle, int process_handle, char *log_file, int timeout);

/**
 * @name    bmcpu_sync_time
 * @brief   Sync device cpu time with host
 * @ingroup bmlib_log
 *
 * @param [in]  handle          The device handle
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bmcpu_sync_time(bm_handle_t handle);

/*******************trace and profile releated functions **********************/
struct bm_heap_stat {
  unsigned int mem_total;
  unsigned int mem_avail;
  unsigned int mem_used;
};

typedef struct bm_heap_stat_byte {
  unsigned int  heap_id;
  unsigned long long mem_total;
  unsigned long long mem_avail;
  unsigned long long mem_used;
  unsigned long long mem_start_addr;
} bm_heap_stat_byte_t;

typedef struct bm_dev_stat {
  int mem_total;
  int mem_used;
  int tpu_util;
  int heap_num;
  struct bm_heap_stat heap_stat[4];
} bm_dev_stat_t;

/**
 * @name    bm_get_stat
 * @brief   To get the stat data at the moment
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [out] profile The result stat data
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_stat(bm_handle_t handle, bm_dev_stat_t *stat);

/**
 * @name    bm_get_gmem_heap_id
 * @brief   To get the heap id of allocated global memory
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  pmem The allocted global memory
 * @param [out] heapid The result of get heap id
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */

DECL_EXPORT bm_status_t bm_get_gmem_heap_id(bm_handle_t handle, bm_device_mem_t *pmem, unsigned int *heapid);

/**
 * @name    sg_get_gmem_heap_id
 * @brief   To get the heap id of allocated global memory
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  pmem The allocted global memory
 * @param [out] heapid The result of get heap id
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */

DECL_EXPORT bm_status_t sg_get_gmem_heap_id(bm_handle_t handle, sg_device_mem_t *pmem, unsigned int *heapid);

/**
 * @name    bm_get_gmem_heap_id_u64
 * @brief   To get the heap id of allocated global memory
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  pmem The allocted global memory
 * @param [out] heapid The result of get heap id
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_gmem_heap_id_u64(bm_handle_t handle, bm_device_mem_u64_t *pmem, unsigned int *heapid);

/**
 * @name    bm_get_gmem_total_heap_num
 * @brief   To get the total heap num of global memory
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  heap_num The result of get total num
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_gmem_total_heap_num(bm_handle_t handle, unsigned int *heap_num);

/**
 * @name    bm_get_gmem_heap_stat_byte_by_id
 * @brief   To get the heap stat by heap id
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  heap_id The heap index to get heap status
 * @param [out] pheap_byte The result of get heap status
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_gmem_heap_stat_byte_by_id(bm_handle_t handle, bm_heap_stat_byte_t *pheap_byte, unsigned int heap_id);

DECL_EXPORT bm_status_t bm_load_firmware(
        bm_handle_t  handle,
        const char  *firmware_tcm,
        const char  *firmware_ddr);

#define bmkernel_load_firmware okkernel_load_firmware
DECL_EXPORT bm_status_t okkernel_load_firmware(
        bm_handle_t  handle,
        const char  *firmware_tcm,
        const char  *firmware_ddr);

DECL_EXPORT bm_status_t okkernel_launch_async(
        bm_handle_t   handle,
        const char   *func_name,
        const void   *args,
        unsigned int  size);

DECL_EXPORT bm_status_t okkernel_launch_sync(
        bm_handle_t   handle,
        const char   *func_name,
        const void   *args,
        unsigned int  size);

DECL_EXPORT bm_status_t tpu_kernel_launch_sync(
        bm_handle_t   handle,
        const char   *func_name,
        const void   *args,
        unsigned int  size);


DECL_EXPORT bm_status_t okkernel_sync(bm_handle_t handle);

/**
 * @name    bmkernel_launch
 * @brief   send api to device and launch function
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  api cmd struct pointer
 * @param [in]  api cmd length
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bmkernel_launch(bm_handle_t handle, const void *args,
                            unsigned int size);

/**
 * @name    bmkernel_load_lookup_table
 * @brief   load lookup table to l2-sram
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle  The device handle
 * @param [in]  table which loaded to l2-sram
 * @param [in]  table size
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bmkernel_load_lookup_table(bm_handle_t handle, const void* table, unsigned int size);

/*******************device management api functions ********************************************/
/**
 * @name    bm_get_tpu_current
 * @brief   get tpu current
 * @ingroup bmlib_runtime
 *
 * @param [in]   handle     The device handle
 * @param [out]  tpuc(mA)   The pointer for tpu current
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_tpu_current(bm_handle_t handle, unsigned int *tpuc);

/**
 * @name    bm_get_board_max_power
 * @brief   get board support max power
 * @ingroup bmlib_runtime
 *
 * @param [in]   handle  The device handle
 * @param [out]  maxp    The pointer for maxp
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_board_max_power(bm_handle_t handle, unsigned int *maxp);

/**
 * @name    bm_get_board_power
 * @brief   get board power
 * @ingroup bmlib_runtime
 *
 * @param [in]   handle    The device handle
 * @param [out]  boardp    The pointer for boardp
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_board_power(bm_handle_t handle, unsigned int *boardp);

/**
 * @name    bm_get_fan_speed
 * @brief   get board fan speed
 * @ingroup bmlib_runtime
 *
 * @param [in]   handle The device handle
 * @param [out]  fan    The pointer for fan speed
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_fan_speed(bm_handle_t handle, unsigned int *fan);

/**
 * @name    bm_get_ecc_correct_num
 * @brief   get ecc_correct_num
 * @ingroup device management api
 *
 * @param [in]   handle  The device handle
 * @param [out]  ecc_correct_num
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
#ifdef __linux__
DECL_EXPORT bm_status_t bm_get_ecc_correct_num(bm_handle_t handle, unsigned long *ecc_correct_num);
#else
DECL_EXPORT bm_status_t bm_get_ecc_correct_num(bm_handle_t handle, unsigned long long *ecc_correct_num);
#endif
/**
 * @name    bm_get_12v_atx
 * @brief   get atx_12v
 * @ingroup device management api
 *
 * @param [in]   handle  The device handle
 * @param [out]  atx_12v
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_12v_atx(bm_handle_t handle, int *atx_12v);

/**
 * @name    bm_get_product_sn
 * @brief   get SE5 sn
 * @ingroup device management api
 *
 * @param [out]  product_sn
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_product_sn(char *product_sn);

/**
 * @name    bm_get_sn
 * @brief   get sn
 * @ingroup device management api
 *
 * @param [in]   handle  The device handle
 * @param [out]  sn
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_sn(bm_handle_t handle, char *sn);

/**
 * @name    bm_get_status
 * @brief   get chip status
 * @ingroup device management api
 *
 * @param [in]   handle  The device handle
 * @param [out]  status  The board error status, each bit represents an error state
 *  status == 0x0, borad is nornal, staus > 0, borad is abnormal;
 *  bit0 == 1, tpu is hang
 *  bit1 == 1, pcie link abnormal
 *  bit2 == 1, board temperature is too high
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_status(bm_handle_t handle, int *status);

/**
 * @name    bm_get_tpu_maxclk
 * @brief   get tpu_maxclk
 * @ingroup device management api
 *
 * @param [in]   handle  The device handle
 * @param [out]  tpu_maxclk
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_tpu_maxclk(bm_handle_t handle, unsigned int *tpu_maxclk);

/**
 * @name    bm_get_tpu_minclk
 * @brief   get tpu_minclk
 * @ingroup device management api
 *
 * @param [in]   handle  The device handle
 * @param [out]  tpu_minclk
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_tpu_minclk(bm_handle_t handle, unsigned int *tpu_minclk);

/**
 * @name    bm_get_driver_version
 * @brief   get driver version
 * @ingroup device management api
 *
 * @param [in]   handle The device handle
 * @param [out]  driver_version
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_driver_version(bm_handle_t handle, int *driver_version);

/**
 * @name    bm_get_board_name
 * @brief   get device board name
 * @ingroup device management api
 *
 * @param [in]   handle The device handle
 * @param [out]  board_name
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_board_name(bm_handle_t handle, char *name);

/**
 * @name    bm_get_board_temp
 * @brief   get board temperature
 * @ingroup device management api
 *
 * @param [in]   handle The device handle
 * @param [out]  board_temp
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_board_temp(bm_handle_t handle, unsigned int *board_temp);

/**
 * @name    bm_get_chip_temp
 * @brief   get chip temperature
 * @ingroup device management api
 *
 * @param [in]   handle The device handle
 * @param [out]  chip_temp
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_chip_temp(bm_handle_t handle, unsigned int *chip_temp);

/**
 * @name    bm_get_tpu_power
 * @brief   get TPU power
 * @ingroup device management api
 *
 * @param [in]   handle The device handle
 * @param [out]  tpu_power
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_tpu_power(bm_handle_t handle, float *tpu_power);

/**
 * @name    bm_get_tpu_volt
 * @brief   get TPU voltage
 * @ingroup device management api
 *
 * @param [in]   handle The device handle
 * @param [out]  The tpu current volt
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_tpu_volt(bm_handle_t handle, unsigned int *tpu_volt);

/**
 * @name    bm_get_card_id
 * @brief   get card id
 * @ingroup device management api
 *
 * @param [in]   handle The device handle
 * @param [out]  card_id
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_card_id(bm_handle_t handle, unsigned int *card_id);

/**
 * @name    bm_get_card_num
 * @brief   get card number
 * @ingroup device management api
 *
 * @param [in]   handle The device handle
 * @param [out]  card_id
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_card_num(unsigned int *card_num);

/**
 * @name    bm_get_chip_num_from_card
 * @brief   get chip number and start chip id from card
 * @ingroup device management api
 *
 * @param [in]   handle The device handle
 * @param [out]  chip_num
 * @param [out]  dev_start_index
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_chip_num_from_card(unsigned int card_id, unsigned int *chip_num, unsigned int *dev_start_index);

/**
 * @name    bm_get_dynfreq_status
 * @brief   get chip dynamic freq status
 * @ingroup device management api
 *
 * @param [in]   handle The device handle
 * @param [out]  dynfreq_status
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_dynfreq_status(bm_handle_t handle, int *dynfreq_status);

/**
 * @name    bm_change_dynfreq_status
 * @brief   change(enable/disable) chip dynamic freq status
 * @ingroup device management api
 *
 * @param [in]   handle The device handle
 * @param [in]   new_status
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_change_dynfreq_status(bm_handle_t handle, int new_status);

/**
 * @name    bm_get_tpu_scalar_num
 * @brief   To get the core number of TPU scalar
 * @ingroup bmlib_runtime
 *
 * @param [in] handle    The device handle
 * @param [out] core_num The core number of TPU scalar
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
DECL_EXPORT bm_status_t bm_get_tpu_scalar_num(bm_handle_t handle, unsigned int *core_num);

#define  bm_get_tpu_core_num bm_get_tpu_scalar_num

typedef struct{
	int core_id;
	tpu_kernel_function_t func_id;
	void *param_data;
	unsigned int param_size;
} tpu_launch_param_t;

/**
 * @name    tpu_kernel_launch_async_multi_cores
 * @brief   To launch function with async for multi cores
 * @ingroup bmlib_runtime
 *
 * @param [in]  handle          The device handle
 * @param [in]  param_list      param_list
 * @param [in]  param_num       param_num
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t tpu_kernel_launch_async_multicores(bm_handle_t handle, tpu_launch_param_t *param_list, int param_num);

#if defined(__cplusplus)
}
#endif

#endif /* BM_RUNTIME_H_ */

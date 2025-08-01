#ifndef __HOST_TEST_UTILS_H__
#define __HOST_TEST_UTILS_H__
#include "common.h"
#include "tpu_defs.h"
#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tpuDNN.h>
#include <vector>
#if defined(__bm1684x__) || defined(__bm1688__) || defined(__mars3__) ||       \
    defined(__bm1684xe__)
#include "bmlib_runtime.h"
#include "common_util.h"
#elif defined(__bm1690__) || defined(__sg2262__) || defined(__sg2260e__)
#include <tpu_fp16.h>
#else
#include <tpu_fp16.h>
#endif
#include <cast.h>
 
#if defined(__sg2260e__) || defined(__sg2262__)
#define array4_t dim4
#endif
 
int DtypeSize(int dtype) {
  int size = 1;
  if (dtype == DT_INT8 || dtype == DT_UINT8)
    size = 1;
  else if (dtype == DT_INT16 || dtype == DT_UINT16 || dtype == DT_FP16 ||
           dtype == DT_BFP16)
    size = 2;
  else if (dtype == DT_FP32 || dtype == DT_INT32 || dtype == DT_UINT32)
    size = 4;
  else if (dtype == DT_INT4 || dtype == DT_UINT4)
    size = 1; // 4-bit types are packed, but for simplicity we use 1 byte per element
  return size;
}
 
void convert_src_to_float(std::vector<float> &dst, void *data, size_t ele_size,
                          int dtype) {
  dst.resize(ele_size);
  for (size_t i = 0; i < ele_size; ++i) {
    if (dtype == DT_FP32)
      dst[i] = reinterpret_cast<float *>(data)[i];
    else if (dtype == DT_INT8)
      dst[i] = reinterpret_cast<char *>(data)[i];
    else if (dtype == DT_UINT8)
      dst[i] = reinterpret_cast<unsigned char *>(data)[i];
    else if (dtype == DT_INT32)
      dst[i] = reinterpret_cast<int *>(data)[i];
    else if (dtype == DT_UINT32)
      dst[i] = reinterpret_cast<u32 *>(data)[i];
    else if (dtype == DT_UINT16)
      dst[i] = reinterpret_cast<u16 *>(data)[i];
    else if (dtype == DT_INT16)
      dst[i] = reinterpret_cast<short *>(data)[i];
    else if (dtype == DT_FP16)
      dst[i] = fp16_to_fp32(reinterpret_cast<fp16 *>(data)[i]).fval;
    else if (dtype == DT_BFP16)
      dst[i] = bf16_to_fp32(reinterpret_cast<bf16 *>(data)[i]).fval;
    else if (dtype == DT_FP8E4M3 || dtype == DT_FP8E5M2) {
      dst[i] = fp8_to_fp32(reinterpret_cast<uint8_t *>(data)[i],
                           dtype == DT_FP8E5M2);
    } else if (dtype == DT_INT4) {
      dst[i] = static_cast<float>(reinterpret_cast<int8_t *>(data)[i]);
    } else if (dtype == DT_UINT4) {
      dst[i] = static_cast<float>(reinterpret_cast<uint8_t *>(data)[i]);
    } else {
      printf("dtype not supported:%d\n", dtype);
      assert(0);
    }
  }
}
void convert_float_to_dst(void *data, std::vector<float> &numbers,
                          size_t ele_size, int dtype) {
  for (size_t i = 0; i < ele_size; ++i) {
    if (dtype == DT_FP32)
      reinterpret_cast<float *>(data)[i] = numbers[i];
    else if (dtype == DT_INT8)
      reinterpret_cast<int8_t *>(data)[i] = static_cast<int8_t>(numbers[i]);
    else if (dtype == DT_UINT8)
      reinterpret_cast<uint8_t *>(data)[i] = static_cast<uint8_t>(numbers[i]);
    else if (dtype == DT_INT32)
      reinterpret_cast<int *>(data)[i] = static_cast<int>(numbers[i]);
    else if (dtype == DT_UINT32)
      reinterpret_cast<uint32_t *>(data)[i] = static_cast<uint32_t>(numbers[i]);
    else if (dtype == DT_UINT16)
      reinterpret_cast<uint16_t *>(data)[i] = static_cast<uint16_t>(numbers[i]);
    else if (dtype == DT_INT16)
      reinterpret_cast<int16_t *>(data)[i] = static_cast<int16_t>(numbers[i]);
    else if (dtype == DT_FP16)
      reinterpret_cast<fp16 *>(data)[i] = fp32_to_fp16(*(fp32 *)&numbers[i]);
    else if (dtype == DT_BFP16)
      reinterpret_cast<bf16 *>(data)[i] = fp32_to_bf16(*(fp32 *)&numbers[i]);
    else if (dtype == DT_FP8E4M3 || dtype == DT_FP8E5M2) {
      reinterpret_cast<uint8_t *>(data)[i] = fp32_to_fp8(
          numbers[i], dtype == DT_FP8E5M2, true, ROUND_HALF_TO_EVEN);
    } else if (dtype == DT_INT4) {
      int8_t value = static_cast<int8_t>(std::round(numbers[i]));
      value = std::max(static_cast<int8_t>(-8), std::min(static_cast<int8_t>(7), value));
      reinterpret_cast<int8_t *>(data)[i] = value;
    } else if (dtype == DT_UINT4) {
      uint8_t value = static_cast<uint8_t>(std::round(numbers[i]));
      value = std::max(static_cast<uint8_t>(0), std::min(static_cast<uint8_t>(15), value));
      reinterpret_cast<uint8_t *>(data)[i] = value;
    } else {
      printf("dtype not supported:%d\n", dtype);
      assert(0);
    }
  }
}
 
void convert_data_type(void *dst_data, void *src_data, int ele_size,
                       int dst_dtype, int src_dtype) {
  if (src_dtype == dst_dtype) {
    memcpy(dst_data, src_data, ele_size * DtypeSize(dst_dtype));
  } else {
    std::vector<float> numbers;
    convert_src_to_float(numbers, src_data, ele_size, src_dtype);
    convert_float_to_dst(dst_data, numbers, ele_size, dst_dtype);
  }
}
 
void load_file(std::string file_path, char *data, size_t data_size,
               size_t ele_size, int dtype, int file_dtype) {
  std::ifstream infile(file_path, std::ios::binary);
  if (!infile.good()) {
    printf("Error! Can't open data file %s!\n", file_path.c_str());
    exit(1);
  }
  infile.seekg(0, std::ios::end);
  size_t file_size = infile.tellg();
  size_t file_ele_size = file_size / DtypeSize(file_dtype);
  if (file_ele_size != ele_size) {
    printf("file_ele_size != ele_size:%ld != %ld\n", file_ele_size, ele_size);
    exit(1);
  }
 
  infile.seekg(0, std::ios::beg);
  std::vector<char> tmp_data;
  tmp_data.resize(file_size);
  infile.read(tmp_data.data(), file_size);
  convert_data_type(data, tmp_data.data(), ele_size, dtype, file_dtype);
 
  printf("read %s success\n", file_path.c_str());
}
 
void dump_data(std::string dir, std::string func_name, std::string pre,
               std::string end, int idx, char *data, size_t data_size,
               int dtype) {
  size_t element_size = data_size / DtypeSize(dtype);
  std::vector<float> out_datas;
  convert_src_to_float(out_datas, data, element_size, dtype);
 
  std::stringstream ss;
  ss << dir << "/" << func_name << pre << idx << end;
  std::ofstream outfile(ss.str(), std::ios::binary);
  outfile.write((char *)out_datas.data(), sizeof(float) * out_datas.size());
  printf("dump %s success\n", ss.str().c_str());
}
 
void rand_data(char *data, dim4 shape, dim4 stride, dim4 offset, float min_val,
               float max_val, int dtype) {
  int64_t calcOffset = offset.n * stride.n + offset.c * stride.c +
                       offset.h * stride.h + offset.w;
 
  std::random_device rd;
  std::mt19937 e(rd());
  std::uniform_real_distribution<float> u(min_val, max_val);
  for (int n = 0; n < shape.n; ++n) {
    for (int c = 0; c < shape.c; ++c) {
      for (int h = 0; h < shape.h; ++h) {
        for (int w = 0; w < shape.w; ++w) {
          int index = n * stride.n + c * stride.c + h * stride.h +
                      w * stride.w + calcOffset;
          if (dtype == DT_FP32) {
            float value = static_cast<float>(u(e));
            reinterpret_cast<float *>(data)[index] = value;
          } else if (dtype == DT_INT8) {
            char value = static_cast<char>(u(e));
            reinterpret_cast<char *>(data)[index] = value;
          } else if (dtype == DT_UINT8) {
            unsigned char value = static_cast<unsigned char>(u(e));
            reinterpret_cast<unsigned char *>(data)[index] = value;
          } else if (dtype == DT_INT32) {
            int value = static_cast<int>(u(e));
            reinterpret_cast<int *>(data)[index] = value;
          } else if (dtype == DT_UINT32) {
            unsigned int value = static_cast<unsigned int>(u(e));
            reinterpret_cast<unsigned int *>(data)[index] = value;
          } else if (dtype == DT_UINT16) {
            unsigned short value = static_cast<unsigned short>(u(e));
            reinterpret_cast<unsigned short *>(data)[index] = value;
          } else if (dtype == DT_INT16) {
            short value = static_cast<short>(u(e));
            reinterpret_cast<short *>(data)[index] = value;
          } else if (dtype == DT_FP16) {
            float rand_value = u(e);
            fp16 value = fp32_to_fp16(*(fp32 *)&rand_value);
            reinterpret_cast<fp16 *>(data)[index] = value;
          } else if (dtype == DT_BFP16) {
            float rand_value = u(e);
            bf16 value = fp32_to_bf16(*(fp32 *)&rand_value);
            reinterpret_cast<bf16 *>(data)[index] = value;
          } else if (dtype == DT_FP8E4M3 || dtype == DT_FP8E5M2) {
            float rand_value = u(e);
            uint8_t value = fp32_to_fp8(rand_value, dtype == DT_FP8E5M2, true,
                                        ROUND_HALF_TO_EVEN);
            reinterpret_cast<uint8_t *>(data)[index] = value;
          } else if (dtype == DT_INT4) {
            // INT4: range -8 to 7, stored as int8_t
            int8_t value = static_cast<int8_t>(std::round(u(e)));
            value = std::max(static_cast<int8_t>(-8), std::min(static_cast<int8_t>(7), value));
            reinterpret_cast<int8_t *>(data)[index] = value;
          } else if (dtype == DT_UINT4) {
            // UINT4: range 0 to 15, stored as uint8_t
            uint8_t value = static_cast<uint8_t>(std::round(u(e)));
            value = std::max(static_cast<uint8_t>(0), std::min(static_cast<uint8_t>(15), value));
            reinterpret_cast<uint8_t *>(data)[index] = value;
          } else {
            printf("dtype not supported:%d\n", dtype);
            assert(0);
          }
        }
      }
    }
  }
}
 
void rand_data(char *data, dim4 shape, dim4 stride, float min_val,
               float max_val, int dtype) {
  dim4 offset = {0, 0, 0, 0};
  rand_data(data, shape, stride, offset, min_val, max_val, dtype);
}
 
void rand_data(char *data, dim4 shape, float min_val, float max_val,
               int dtype) {
  dim4 stride;
  stride.n = shape.c * shape.h * shape.w;
  stride.c = shape.h * shape.w;
  stride.h = shape.w;
  stride.w = 1;
  dim4 offset = {0, 0, 0, 0};
  rand_data(data, shape, stride, offset, min_val, max_val, dtype);
}
 
tpudnnHandle_t t_handle;
void tpu_sync() { tpudnnSync(t_handle); }
#if defined(__bm1684x__) || defined(__bm1688__) || defined(__mars3__) ||       \
    defined(__bm1684xe__)
bool MallocWrap(bm_handle_t &bm_handle, bm_device_mem_t *dev_mem,
                unsigned long long *host_mem, size_t size) {
  bm_status_t status = bm_malloc_device_byte(bm_handle, dev_mem, size);
  assert(BM_SUCCESS == status);
#ifdef SOC_MODE
  status = bm_mem_mmap_device_mem(bm_handle, dev_mem, host_mem);
  assert(BM_SUCCESS == status);
#else
  *host_mem = (unsigned long long)malloc(size);
#endif
  return BM_SUCCESS == status;
}
 
void FreeWrap(bm_handle_t &bm_handle, bm_device_mem_t *dev_mem,
              void *host_mem) {
#ifdef SOC_MODE
  bm_status_t status = bm_mem_unmap_device_mem(
      bm_handle, host_mem, bm_mem_get_device_size(*dev_mem));
  assert(BM_SUCCESS == status);
#else
  free(host_mem);
#endif
  bm_free_device(bm_handle, *dev_mem);
}
 
inline void MemcpyS2D(bm_handle_t &bm_handle, bm_device_mem_t *dev_mem,
                      void *host_mem, size_t size, unsigned int offset = 0) {
  if (size == 0)
    return;
#ifdef SOC_MODE
  bm_mem_flush_partial_device_mem(bm_handle, dev_mem, offset, size);
#else
  bm_memcpy_s2d_partial_offset(bm_handle, *dev_mem, (int8_t *)host_mem + offset,
                               size, offset);
#endif
}
 
inline void MemcpyD2S(bm_handle_t &bm_handle, bm_device_mem_t *dev_mem,
                      void *host_mem, size_t size, unsigned int offset = 0) {
  if (size == 0)
    return;
#ifdef SOC_MODE
  bm_mem_invalidate_partial_device_mem(bm_handle, dev_mem, offset, size);
#else
  bm_memcpy_d2s_partial_offset(bm_handle, (int8_t *)host_mem + offset, *dev_mem,
                               size, offset);
#endif
}
 
#include "kernel_module_data.h"
bm_handle_t handle;
#if defined(__bm1688__)
tpu_kernel_module_t tpu_module[2];
#else
tpu_kernel_module_t tpu_module;
#endif
 
struct mem_t {
  u64 dev_mem = 0;
  char *host_mem = nullptr;
  bm_device_mem_t dev_mem_data;
  dim4 shape;
  int dtype;
  size_t data_size;
  size_t ele_size;
  ~mem_t() {
    if (host_mem || dev_mem) {
      FreeWrap(handle, &dev_mem_data, host_mem);
    }
  }
};
 
static mem_t alloc_mem(dim4 &shape, int dtype) {
  mem_t mem;
  mem.ele_size = static_cast<size_t>(shape.n) * shape.c * shape.h * shape.w;
  size_t size = mem.ele_size * DtypeSize(dtype);
  mem.data_size = size;
  mem.dtype = dtype;
  mem.shape = shape;
  MallocWrap(handle, &mem.dev_mem_data, (u64 *)&mem.host_mem, mem.data_size);
  memset((char *)mem.host_mem, 0x00, mem.data_size);
  mem.dev_mem = bm_mem_get_device_addr(mem.dev_mem_data);
  return mem;
}
static mem_t alloc_rand_mem(dim4 &shape, int dtype, float min, float max) {
  mem_t mem;
  mem.ele_size = static_cast<size_t>(shape.n) * shape.c * shape.h * shape.w;
  size_t size = mem.ele_size * DtypeSize(dtype);
  mem.data_size = size;
  mem.dtype = dtype;
  mem.shape = shape;
  MallocWrap(handle, &mem.dev_mem_data, (u64 *)&mem.host_mem, mem.data_size);
  rand_data(mem.host_mem, shape, min, max, dtype);
  mem.dev_mem = bm_mem_get_device_addr(mem.dev_mem_data);
  return mem;
}
 
static mem_t alloc_mem_copy_from(mem_t &other) {
  mem_t mem;
  mem.shape = other.shape;
  mem.dtype = other.dtype;
  mem.ele_size = other.ele_size;
  mem.data_size = other.data_size;
  mem.host_mem = new char[other.data_size];
  MallocWrap(handle, &mem.dev_mem_data, (u64 *)&mem.host_mem, mem.data_size);
  memcpy(mem.host_mem, other.host_mem, other.data_size);
  mem.dev_mem = bm_mem_get_device_addr(mem.dev_mem_data);
  return mem;
}
 
static mem_t copy_mem(mem_t &dst, mem_t &src) {
  memcpy(dst.host_mem, src.host_mem, src.data_size);
  return dst;
}
 
void D2S(mem_t &mem) {
  MemcpyD2S(handle, &mem.dev_mem_data, mem.host_mem, mem.data_size);
}
void S2D(mem_t &mem) {
  MemcpyS2D(handle, &mem.dev_mem_data, mem.host_mem, mem.data_size);
}
 
int init_device() {
  bm_status_t ret = BM_SUCCESS;
  int devid = 0;
  const char *devid_env = getenv("PPL_DEVID");
  if (devid_env) {
    devid = atoi(devid_env);
  }
  ret = bm_dev_request(&handle, devid);
  if (ret != BM_SUCCESS) {
    printf("init device failed\n");
    return -1;
  }
  printf("init device success\n");
  const unsigned int *p = kernel_module_data;
  size_t length = sizeof(kernel_module_data);
#if defined(__bm1688__)
  for (int i = 0; i < 2; ++i) {
    tpu_module[i] = tpu_kernel_load_module(handle, (const char *)p, length);
    if (!tpu_module[i]) {
      printf("load module failed\n");
      return -1;
    }
  }
#else
  tpu_module = tpu_kernel_load_module(handle, (const char *)p, length);
  if (!tpu_module) {
    printf("load module failed\n");
    return -1;
  }
#endif
  printf("load module success!\n");
  t_handle = tpudnnHandleFromStream(devid, handle, tpu_module);
  return 0;
}
 
void release_device() {
#if defined(__bm1688__)
  tpu_kernel_free_module(handle, tpu_module[0]);
  tpu_kernel_free_module(handle, tpu_module[1]);
#else
  tpu_kernel_free_module(handle, tpu_module);
#endif
  bm_dev_free(handle);
  tpudnnDestroy(t_handle);
}
void free_mem(mem_t &mem) {
  if (mem.host_mem || mem.dev_mem) {
    FreeWrap(handle, &mem.dev_mem_data, mem.host_mem);
    mem.host_mem = nullptr;
    mem.dev_mem = 0;
  }
}
#elif defined(__bm1690__) || defined(__sg2262__) || defined(__sg2260e__)
tpuRtStream_t stream;
tpuRtKernelModule_t tpu_module;
 
struct mem_t {
  u64 dev_mem = 0;
  char *host_mem = nullptr;
  dim4 shape;
  int dtype;
  size_t data_size;
  size_t ele_size;
  ~mem_t() {
    if (host_mem) {
      delete[] host_mem;
    }
    if (dev_mem) {
      tpuRtFree((void **)&dev_mem, 0);
    }
  }
};
 
static mem_t alloc_mem(dim4 &shape, int dtype) {
  mem_t mem;
  mem.ele_size = static_cast<size_t>(shape.n) * shape.c * shape.h * shape.w;
  size_t size = mem.ele_size * DtypeSize(dtype);
  char *data = new char[size];
  memset(data, 0x00, size);
  mem.host_mem = data;
  mem.dev_mem = 0;
  mem.data_size = size;
  mem.dtype = dtype;
  mem.shape = shape;
  tpuRtMalloc((void **)(&mem.dev_mem), size, 0);
  memset((char *)mem.host_mem, 0x00, size);
  return mem;
}
static mem_t alloc_rand_mem(dim4 &shape, int dtype, float min, float max) {
  mem_t mem;
  mem.ele_size = static_cast<size_t>(shape.n) * shape.c * shape.h * shape.w;
  size_t size = mem.ele_size * DtypeSize(dtype);
  char *data = new char[size];
  memset(data, 0x00, size);
  mem.host_mem = data;
  mem.dev_mem = 0;
  mem.data_size = size;
  mem.dtype = dtype;
  mem.shape = shape;
  tpuRtMalloc((void **)(&mem.dev_mem), size, 0);
  rand_data(mem.host_mem, shape, min, max, dtype);
  return mem;
}
static mem_t alloc_mem_copy_from(mem_t &other) {
  mem_t mem;
  mem.shape = other.shape;
  mem.dtype = other.dtype;
  mem.ele_size = other.ele_size;
  mem.data_size = other.data_size;
  mem.host_mem = new char[other.data_size];
  memcpy(mem.host_mem, other.host_mem, other.data_size);
  tpuRtMalloc((void **)(&mem.dev_mem), other.data_size, 0);
  return mem;
}
 
static mem_t copy_mem(mem_t &dst, mem_t &src) {
  memcpy(dst.host_mem, src.host_mem, src.data_size);
  return dst;
}
 
void D2S(mem_t &mem) {
  tpuRtMemcpyD2S(mem.host_mem, (void *)mem.dev_mem, mem.data_size);
}
void S2D(mem_t &mem) {
  tpuRtMemcpyS2D((void *)mem.dev_mem, mem.host_mem, mem.data_size);
}
 
int init_device() {
  int devid = 0;
  const char *devid_env = getenv("PPL_DEVID");
  if (devid_env) {
    devid = atoi(devid_env);
  }
  tpuRtStatus_t ret;
  ret = tpuRtInit();
  if (ret != tpuRtSuccess) {
    printf("init device failed\n");
    return -1;
  }
  printf("init device success\n");
  tpuRtSetDevice(devid);
  tpuRtStreamCreate(&stream);
  auto kernel_dir = getenv("PPL_KERNEL_PATH");
  if (!kernel_dir) {
    printf("Please set env PPL_KERNEL_PATH to libkernel.so path\n");
    return -1;
  }
  tpu_module = tpuRtKernelLoadModuleFile(kernel_dir, stream);
  if (NULL == tpu_module) {
    printf("load module failed\n");
    return -2;
  }
  printf("load module success!\n");
  t_handle = tpudnnHandleFromStream(devid, stream, tpu_module);
  return 0;
}
 
void release_device() {
  tpuRtKernelUnloadModule(tpu_module, stream);
  tpuRtStreamSynchronize(stream);
  tpuRtStreamDestroy(stream);
  tpudnnDestroy(t_handle);
}
 
void free_mem(mem_t &mem) {
  if (mem.host_mem) {
    delete[] mem.host_mem;
    mem.host_mem = nullptr;
  }
  if (mem.dev_mem) {
    tpuRtFree((void **)&mem.dev_mem, 0);
    mem.dev_mem = 0;
  }
}
#endif
 
#endif

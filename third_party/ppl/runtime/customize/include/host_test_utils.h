#ifndef __HOST_TEST_UTILS_H__
#define __HOST_TEST_UTILS_H__
#include "cnpy.h"
#include "common.h"
#include "tpu_defs.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#if defined(__bm1684x__) || defined(__bm1688__) || defined(__mars3__) ||       \
    defined(__bm1684xe__)
#include "bmlib_runtime.h"
#include "common_util.h"
#elif defined(__bm1690__) || defined(__sg2262__)
#include <tpu_fp16.h>
#else
#include <tpu_fp16.h>
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
  return size;
}

template <typename T>
void transport(char *data, std::vector<float> &numbers, size_t type_size,
               size_t data_size) {
  size_t bytes_to_write = std::min(numbers.size() * type_size, data_size);
  for (size_t i = 0; i < numbers.size() && (i * type_size) < bytes_to_write;
       ++i) {
    char *new_address = data + i * type_size;
    T value = static_cast<T>(numbers[i]);
    memcpy(new_address, &value, type_size);
  }
}

template <>
void transport<bf16>(char *data, std::vector<float> &numbers, size_t type_size,
                     size_t data_size) {
  size_t bytes_to_write = std::min(numbers.size() * type_size, data_size);
  for (size_t i = 0; i < numbers.size() && (i * type_size) < bytes_to_write;
       ++i) {
    char *new_address = data + i * type_size;
    bf16 value = fp32_to_bf16(*(fp32 *)&numbers[i]);
    memcpy(new_address, &value, type_size);
  }
}

template <>
void transport<fp16>(char *data, std::vector<float> &numbers, size_t type_size,
                     size_t data_size) {
  size_t bytes_to_write = std::min(numbers.size() * type_size, data_size);
  for (size_t i = 0; i < numbers.size() && (i * type_size) < bytes_to_write;
       ++i) {
    char *new_address = data + i * type_size;
    fp16 value = fp32_to_fp16(*(fp32 *)&numbers[i]);
    memcpy(new_address, &value, type_size);
  }
}

void transport(char *data, std::vector<float> &numbers, size_t type_size,
               size_t data_size, int dtype) {
  switch (dtype) {
  case DT_FP32: {
    transport<float>(data, numbers, type_size, data_size);
    break;
  }
  case DT_INT8: {
    transport<char>(data, numbers, type_size, data_size);
    break;
  }
  case DT_UINT8: {
    transport<unsigned char>(data, numbers, type_size, data_size);
    break;
  }
  case DT_INT32: {
    transport<int>(data, numbers, type_size, data_size);
    break;
  }
  case DT_UINT32: {
    transport<u32>(data, numbers, type_size, data_size);
    break;
  }
  case DT_UINT16: {
    transport<u16>(data, numbers, type_size, data_size);
    break;
  }
  case DT_INT16: {
    transport<short>(data, numbers, type_size, data_size);
    break;
  }
  case DT_FP16: {
    transport<fp16>(data, numbers, type_size, data_size);
    break;
  }
  case DT_BFP16: {
    transport<bf16>(data, numbers, type_size, data_size);
    break;
  }
  default:
    std::cerr << "Unsupported data type: " << dtype << std::endl;
    assert(false);
  }
}

void load_file(std::string file_path, char *data, size_t data_size,
               size_t file_size, int dtype, int file_dtype) {
  std::stringstream ss;
  ss << file_path;
  std::ifstream infile(ss.str(), std::ios::binary);
  if (!infile.good()) {
    printf("Error! Can't open data file %s!\n", ss.str().c_str());
    exit(1);
  }

  infile.seekg(0, std::ios::beg);
  infile.read(data, file_size);

  if (dtype != file_dtype) {
    std::vector<float> numbers;
    size_t element_size = file_size / DtypeSize(file_dtype);
    for (size_t i = 0; i < element_size; ++i) {
      float value = 0.0l;
      if (file_dtype == DT_FP32)
        value = reinterpret_cast<float *>(data)[i];
      else if (file_dtype == DT_INT8)
        value = reinterpret_cast<char *>(data)[i];
      else if (file_dtype == DT_UINT8)
        value = reinterpret_cast<unsigned char *>(data)[i];
      else if (file_dtype == DT_INT32)
        value = reinterpret_cast<int *>(data)[i];
      else if (file_dtype == DT_UINT32)
        value = reinterpret_cast<u32 *>(data)[i];
      else if (file_dtype == DT_UINT16)
        value = reinterpret_cast<u16 *>(data)[i];
      else if (file_dtype == DT_INT16)
        value = reinterpret_cast<short *>(data)[i];
      else if (file_dtype == DT_FP16)
        value = fp16_to_fp32(reinterpret_cast<fp16 *>(data)[i]).fval;
      else if (file_dtype == DT_BFP16)
        value = bf16_to_fp32(reinterpret_cast<bf16 *>(data)[i]).fval;
      else {
        printf("dtype not supported:%d\n", file_dtype);
        assert(0);
      }
      // printf("   %f   \n", value);
      numbers.emplace_back(value);
    }

    transport(data, numbers, DtypeSize(dtype), data_size, dtype);
  }

  printf("read %s success\n", ss.str().c_str());
}

void dump_data(std::string dir, std::string func_name, std::string pre,
               std::string end, int idx, char *data, size_t data_size,
               int dtype) {
  std::stringstream ss;
  ss << dir << "/" << func_name << pre << idx << end;
  std::ofstream outfile(ss.str(), std::ios::binary);
  size_t element_size = data_size / DtypeSize(dtype);
  for (size_t i = 0; i < element_size; ++i) {
    float value = 0.0l;
    if (dtype == DT_FP32)
      value = reinterpret_cast<float *>(data)[i];
    else if (dtype == DT_INT8)
      value = reinterpret_cast<char *>(data)[i];
    else if (dtype == DT_UINT8)
      value = reinterpret_cast<unsigned char *>(data)[i];
    else if (dtype == DT_INT32)
      value = reinterpret_cast<int *>(data)[i];
    else if (dtype == DT_UINT32)
      value = reinterpret_cast<u32 *>(data)[i];
    else if (dtype == DT_UINT16)
      value = reinterpret_cast<u16 *>(data)[i];
    else if (dtype == DT_INT16)
      value = reinterpret_cast<short *>(data)[i];
    else if (dtype == DT_FP16)
      value = fp16_to_fp32(reinterpret_cast<fp16 *>(data)[i]).fval;
    else if (dtype == DT_BFP16)
      value = bf16_to_fp32(reinterpret_cast<bf16 *>(data)[i]).fval;
    else {
      printf("dtype not supported:%d\n", dtype);
      assert(0);
    }
    outfile.write((char *)&value, sizeof(float));
  }
  printf("dump %s success\n", ss.str().c_str());
}

void rand_data(std::string dir, std::string func_name, int idx, char *data,
               size_t data_size, dim4 *shape, dim4 *stride, dim4 *offset,
               float min_val, float max_val, int dtype) {
  int64_t calcOffset = offset->n * stride->n + offset->c * stride->c +
                       offset->h * stride->h + offset->w;

  std::random_device rd;
  std::mt19937 e(rd());
  std::uniform_real_distribution<float> u(min_val, max_val);
  for (int n = 0; n < shape->n; ++n) {
    for (int c = 0; c < shape->c; ++c) {
      for (int h = 0; h < shape->h; ++h) {
        for (int w = 0; w < shape->w; ++w) {
          int index = n * stride->n + c * stride->c + h * stride->h +
                      w * stride->w + calcOffset;
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
          } else {
            printf("dtype not supported:%d\n", dtype);
            assert(0);
          }
        }
      }
    }
  }
  std::stringstream ss;
  ss << dir << "/" << func_name << "_" << idx << ".in";
  printf("rand %s success\n", ss.str().c_str());
}

void rand_data(std::string dir, std::string func_name, int idx, char *data,
               size_t data_size, dim4 *shape, dim4 *stride, float min_val,
               float max_val, int dtype) {
  dim4 offset = {0, 0, 0, 0};
  rand_data(dir, func_name, idx, data, data_size, shape, stride, &offset,
            min_val, max_val, dtype);
}

void rand_data(std::string dir, std::string func_name, int idx, char *data,
               size_t data_size, dim4 *shape, float min_val, float max_val,
               int dtype) {
  dim4 stride;
  stride.n = shape->c * shape->h * shape->w;
  stride.c = shape->h * shape->w;
  stride.h = shape->w;
  stride.w = 1;
  dim4 offset = {0, 0, 0, 0};
  rand_data(dir, func_name, idx, data, data_size, shape, &stride, &offset,
            min_val, max_val, dtype);
}

void rand_data(char *data, size_t data_size, dim4 &shape, float min_val,
               float max_val, int dtype) {
  std::random_device rd;
  std::mt19937 e(rd());
  std::uniform_real_distribution<float> u(min_val, max_val);
  dim4 stride = {shape.c * shape.h * shape.w, shape.h * shape.w, shape.w, 1};
  for (int n = 0; n < shape.n; ++n) {
    for (int c = 0; c < shape.c; ++c) {
      for (int h = 0; h < shape.h; ++h) {
        for (int w = 0; w < shape.w; ++w) {
          int index = n * stride.n + c * stride.c + h * stride.h + w * stride.w;
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
          } else {
            printf("dtype not supported:%d\n", dtype);
            assert(0);
          }
        }
      }
    }
  }
}
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
  u64 dev_mem;
  char *host_mem;
  bm_device_mem_t dev_mem_data;
  dim4 shape;
  int dtype;
  size_t data_size;
};

static mem_t alloc_rand_mem(dim4 &shape, int dtype, float min, float max) {
  mem_t mem;
  size_t size = static_cast<size_t>(shape.n) * shape.c * shape.h * shape.w *
                DtypeSize(dtype);
  char *data = new char[size];
  memset(data, 0x00, size);
  mem.host_mem = data;
  mem.data_size = size;
  mem.dtype = dtype;
  mem.shape = shape;
  MallocWrap(handle, &mem.dev_mem_data, (u64 *)&mem.host_mem, size);
  rand_data(data, size, shape, min, max, dtype);
  MemcpyS2D(handle, &mem.dev_mem_data, mem.host_mem, size);
  mem.dev_mem = bm_mem_get_device_addr(mem.dev_mem_data);
  return mem;
}

void D2S(mem_t &mem) {
  MemcpyD2S(handle, &mem.dev_mem_data, mem.host_mem, mem.data_size);
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
}

void free_mem(mem_t &mem) { FreeWrap(handle, &mem.dev_mem_data, mem.host_mem); }
#elif defined(__bm1690__) || defined(__sg2262__)
tpuRtStream_t stream;
tpuRtKernelModule_t tpu_module;

struct mem_t {
  u64 dev_mem;
  char *host_mem;
  dim4 shape;
  int dtype;
  size_t data_size;
};

static mem_t alloc_rand_mem(dim4 &shape, int dtype, float min, float max) {
  mem_t mem;
  size_t size = static_cast<size_t>(shape.n) * shape.c * shape.h * shape.w *
                DtypeSize(dtype);
  char *data = new char[size];
  memset(data, 0x00, size);
  mem.host_mem = data;
  mem.dev_mem = 0;
  mem.data_size = size;
  mem.dtype = dtype;
  mem.shape = shape;
  tpuRtMalloc((void **)(&mem.dev_mem), size, 0);
  rand_data(data, size, shape, min, max, dtype);
  tpuRtMemcpyS2D((void *)mem.dev_mem, mem.host_mem, size);
  return mem;
}

void D2S(mem_t &mem) {
  tpuRtMemcpyD2S(mem.host_mem, (void *)mem.dev_mem, mem.data_size);
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
  return 0;
}

void release_device() {
  tpuRtKernelUnloadModule(tpu_module, stream);
  tpuRtStreamSynchronize(stream);
  ;
  tpuRtStreamDestroy(stream);
}

void free_mem(mem_t &mem) {
  tpuRtFree((void **)&mem.dev_mem, 0);
  delete[] mem.host_mem;
}

#endif

#endif

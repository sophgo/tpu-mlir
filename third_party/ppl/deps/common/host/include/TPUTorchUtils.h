#pragma once
#include <torch/extension.h>
#include <tpuDNN.h>

#define TPU_DEVICE_INDEX_BITS 6
#define TPU_GLOBAL_ADDR_BITS (64 - TPU_DEVICE_INDEX_BITS)
inline tpudnnHandle_t g_handle_t;

inline tpudnnHandle_t tpudnnGetHandle(uint64_t handle_t) {
  if (handle_t)
    g_handle_t = reinterpret_cast<tpudnnHandle_t>(handle_t);
  else
    g_handle_t = tpudnnCreate();
  return g_handle_t;
}

static inline unsigned long long GetAddrByUnifiedAddr ( unsigned long long Addr )
{
  return ( Addr << TPU_DEVICE_INDEX_BITS ) >> TPU_DEVICE_INDEX_BITS;
}

static inline unsigned long long getTensorAddr(const at::Tensor & Tensor) {
  unsigned long long data_ptr = (unsigned long long)Tensor.data_ptr();
  if (Tensor.device().type() == c10::DeviceType::PrivateUse1) {   // DeviceType::TPU == 19
   data_ptr = GetAddrByUnifiedAddr(data_ptr);
  }
  return data_ptr;
}

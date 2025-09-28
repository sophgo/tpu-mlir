#pragma once
#include <torch/extension.h>
#include <tpuDNN.h>
#include <tpuDNNTensor.h>
#include "TPUFormatCastHelper.h"


// constexpr const static func_Sgdnn_t TPUGenerateSgdnnTensor = TPUGenerateDnnTensor<SgdnnTensor_t>;
inline tpudnnHandle_t g_handle_t;
inline tpudnnHandle_t tpudnnGetHandle(uint64_t handle_t) {
  tpudnnHandle_t _handle_t;
  if (handle_t)
    g_handle_t = (tpudnnHandle_t*) handle_t;
  else
    g_handle_t = tpudnnCreate();
  return g_handle_t;
}


#define TPU_DEVICE_INDEX_BITS 6
#define TPU_GLOBAL_ADDR_BITS (64 - TPU_DEVICE_INDEX_BITS)

static inline unsigned long long UnifiedAddr( unsigned long long Addr, int Index)
{
  TORCH_CHECK ( Addr < ( 1UL << TPU_GLOBAL_ADDR_BITS ) );
  return ( ( ( unsigned long long ) Index ) << TPU_GLOBAL_ADDR_BITS ) | Addr;
}

static inline unsigned long long GetDeviceIndexByUnifiedAddr ( unsigned long long Addr )
{
  return Addr >> TPU_GLOBAL_ADDR_BITS;
}

static inline unsigned long long GetAddrByUnifiedAddr ( unsigned long long Addr )
{
  return ( Addr << TPU_DEVICE_INDEX_BITS ) >> TPU_DEVICE_INDEX_BITS;
}

#define define_converter_entry(T, PREFIX, dtype) \
  const static T dtype = PREFIX##_##dtype;

#define define_converter(T, PREFIX)                   \
  define_converter_entry(T, PREFIX, DTYPE_FP32)       \
  define_converter_entry(T, PREFIX, DTYPE_FP16)       \
  define_converter_entry(T, PREFIX, DTYPE_BF16)       \
  define_converter_entry(T, PREFIX, DTYPE_INT64)      \
  define_converter_entry(T, PREFIX, DTYPE_INT32)      \
  define_converter_entry(T, PREFIX, DTYPE_UINT8)      \
  define_converter_entry(T, PREFIX, DTYPE_INT8)       \
  define_converter_entry(T, PREFIX, DTYPE_INT16)      \
  define_converter_entry(T, PREFIX, DTYPE_FP8E4M3)    \
  define_converter_entry(T, PREFIX, DTYPE_UNKNOWN)

template <typename T>
struct dtypes
{
  define_converter(tpudnnDataType_t, TPUDNN)
};

#undef define_converter
#undef define_converter_entry

template <typename T>
static inline T TPUConvertDtype ( caffe2::TypeMeta dtype )
{
  if ( dtype == caffe2::TypeMeta::Make<float>() )
  {
    return dtypes<T>::DTYPE_FP32;
  }
  else if ( dtype == caffe2::TypeMeta::Make<at::Half>() )
  {
    return dtypes<T>::DTYPE_FP16;
  }
  else if ( dtype == caffe2::TypeMeta::Make<at::BFloat16>() )
  {
    return dtypes<T>::DTYPE_BF16;
  }
  else if ( dtype == caffe2::TypeMeta::Make<long>() )
  {
    return dtypes<T>::DTYPE_INT64;
  }
  else if ( dtype == caffe2::TypeMeta::Make<int>() )
  {
    return dtypes<T>::DTYPE_INT32;
  }
  else if ( dtype == caffe2::TypeMeta::Make<bool>() |
            dtype == caffe2::TypeMeta::Make<unsigned char>() ) {
    return dtypes<T>::DTYPE_UINT8;
  }
  else if ( dtype == caffe2::TypeMeta::Make<int8_t>() )
  {
    return dtypes<T>::DTYPE_INT8;
  }
  else if ( dtype == caffe2::TypeMeta::Make<short>() ) {
    return dtypes<T>::DTYPE_INT16;
  }
  else if ( dtype == caffe2::TypeMeta::Make<at::Float8_e4m3fn>() ) {
    return dtypes<T>::DTYPE_FP8E4M3;
  }
  else
  {
    TORCH_CHECK ( false, "Unsupported data type ", dtype );
  }
  return dtypes<T>::DTYPE_UNKNOWN;
}

static inline tpudnnTensor_t TPUGenerateTpudnnTensor(tpudnnHandle_t handle, const at::Tensor & Tensor)
{
  tpudnnTensor_t t = { 0 };
  unsigned long long data_ptr;
  if (at_tpu::StorageDescHelper::IsBaseFormatType(Tensor)) {
    data_ptr = (unsigned long long)Tensor.data_ptr();
    t.dtype =TPUConvertDtype<decltype(t.dtype)>( Tensor.dtype() );
    t.dim = Tensor.dim();
    for ( auto i = 0; i < Tensor.dim(); ++i )
    {
      t.shape[i] = Tensor.size ( i );
      t.stride[i] = Tensor.stride ( i );
    }
  } else {
    TORCH_CHECK(false, "Don't support Tensor is formatcast now.\n");
    // data_ptr = at_tpu::StorageDescHelper::GetDataPtrWithFormat(Tensor);
    // at_tpu::StorageDescHelper::SettpuTensorAttributeWithFormat(Tensor, t);
  }
  if (Tensor.device().type() == c10::DeviceType::PrivateUse1) {   // DeviceType::TPU == 19
    t.addr = reinterpret_cast<decltype(t.addr)>( GetAddrByUnifiedAddr( data_ptr ) );
    t.addr = tpudnnPhysToVirt(handle, (unsigned long long)t.addr);
  }
  return t;
}

static inline std::string GetTensorInfo( const at::Tensor & Tensor )
{
  std::ostringstream Tensor_info;
  if (Tensor.has_storage()){
    auto dtype = Tensor.dtype();
    Tensor_info << "addr : " << Tensor.data_ptr() << ", ";
    Tensor_info << "dtype : " << dtype << ", ";
    if (dtype == caffe2::TypeMeta::Make<float>())
    {
      Tensor_info << "data0 : " << *((float*)(Tensor.cpu().data_ptr())) << ",";
    }
    else if (dtype == caffe2::TypeMeta::Make<c10::Half>())
    {
      Tensor_info << "data0 : " << (float)c10::detail::fp16_ieee_to_fp32_value(((c10::Half*)(Tensor.cpu().data_ptr()))->x) << ",";
    }
    Tensor_info << "size : [";
    for ( auto i = 0; i < Tensor.dim(); ++i )
    {
      Tensor_info << " " << Tensor.size ( i );
    }
    Tensor_info << "], strides : [";
    for ( auto i = 0; i < Tensor.dim(); ++i )
    {
      Tensor_info << " " << Tensor.stride ( i );
    }
    Tensor_info << "];";
  }
  return Tensor_info.str();
}

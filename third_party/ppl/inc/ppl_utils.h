//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022  Technologies Inc.  All rights reserved.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "ppl_defs.h"
#include "ppl_tpu.h"

namespace ppl {

template <typename U, typename V> int align(U numerator, V denominator) {
  return (numerator + denominator - 1) / denominator * denominator;
}

template <typename U, typename V> int div_up(U numerator, V denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <typename U, typename V> auto MAX(U a, V b) {
  return (((a)) > ((b)) ? (a) : (b));
}

template <typename T> void swap(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

template <typename T>
dim4 &get_stride(dim4 &shape, align_mode_t mode, int start_idx = 0) {
  int eu_num = get_eu_num<T>();
  int stride_n = 0;
  int stride_c = 0;
  int stride_h = 0;
  int stride_w = 0;
  if (mode == TPU_ALIGN) {
    stride_h = shape.w;
    stride_c = shape.h * stride_h;
    stride_c = align(stride_c, eu_num);
    stride_n = div_up(start_idx + shape.c, LANE_NUM) * stride_c;
    stride_w = 1;
  } else if (mode == TPU_COMPACT) {
    stride_h = shape.w;
    stride_c = shape.h * stride_h;
    stride_n = div_up(start_idx + shape.c, LANE_NUM) * stride_c;
    stride_w = 1;
  } else if (mode == TPU_ROW_ALIGN) {
    stride_n = div_up(start_idx + shape.c, LANE_NUM) * stride_c;
    stride_c = shape.h * stride_h;
    stride_h = div_up(shape.w, eu_num);
    stride_w = 1;
  } else {
    stride_n = shape.c * shape.h * shape.w;
    stride_c = shape.h * shape.w;
    stride_h = shape.w;
    stride_w = 1;
  }
  dim4 stride = {stride_n, stride_c, stride_h, stride_w};
  return stride;
}

template <typename T> data_type_t convert_dtype() {
  if constexpr (std::is_same_v<T, float>) {
    return DT_FP32;
  } else if constexpr (std::is_same_v<T, int32>) {
    return DT_INT32;
  } else if constexpr (std::is_same_v<T, uint32>) {
    return DT_UINT32;
  } else if constexpr (std::is_same_v<T, int16>) {
    return DT_INT16;
  } else if constexpr (std::is_same_v<T, uint16>) {
    return DT_UINT16;
  } else if constexpr (std::is_same_v<T, int8>) {
    return DT_INT8;
  } else if constexpr (std::is_same_v<T, uint8>) {
    return DT_UINT8;
  } else if constexpr (std::is_same_v<T, int64>) {
    return DT_INT64;
  } else if constexpr (std::is_same_v<T, uint64>) {
    return DT_UINT64;
  } else if constexpr (std::is_same_v<T, bf16>) {
    return DT_BF16;
  } else if constexpr (std::is_same_v<T, fp16>) {
    return DT_FP16;
  } else if constexpr (std::is_same_v<T, fp8e4m3>) {
    return DT_FP8E4M3;
  } else if constexpr (std::is_same_v<T, fp8e5m2>) {
    return DT_FP8E5M2;
  } else if constexpr (std::is_same_v<T, fp20>) {
    return DT_FP20;
  } else if constexpr (std::is_same_v<T, fp4>) {
    return DT_FP4;
  } else {
    return DT_NONE;
  }
}

template <typename T> int dim4_index(const dim4 &d, T index) {
  if (index < 0 && index >= -4)
    index = index + 4;
  if (index == 0)
    return d.n;
  else if (index == 1)
    return d.c;
  else if (index == 2)
    return d.h;
  else if (index == 3)
    return d.w;
  else
    return -1;
}

template <typename dtype, typename dimT>
tensor<dtype> &make_tensor(dimT &block_shape, dimT &real_shape,
                           align_mode_t align_mode = TPU_ALIGN) {
  tensor<dtype> new_tensor(block_shape, align_mode);
  return new_tensor.view(real_shape);
}

template <typename dtype>
gtensor<dtype> &make_l2tensor(dim4 &block_shape, tensor_mode_t mode,
                              dim4 &real_shape) {
  gtensor<dtype> new_tensor(block_shape, mode);
  return new_tensor.view(real_shape);
}

template <typename dtype>
gtensor<dtype> &make_gtensor(dim4 &shape, tensor_mode_t mode, dtype *addr,
                             dim4 &stride) {
  gtensor<dtype> new_tensor(shape, mode, addr);
  return new_tensor.view(shape, stride);
}

template <typename dtype>
gtensor<dtype> &make_gtensor_permute(dim4 &mem_shape, tensor_mode_t mode,
                                     dtype *addr, int order[4]) {
  int mem_stride_n = mem_shape.c * mem_shape.h * mem_shape.w;
  int mem_stride_c = mem_shape.h * mem_shape.w;
  int mem_stride_h = mem_shape.w;
  int mem_stride_w = 1;
  dim4 mem_stride = {mem_stride_n, mem_stride_c, mem_stride_h, mem_stride_w};
  dim4 stride_permute = {
      dim4_index(mem_stride, order[0]), dim4_index(mem_stride, order[1]),
      dim4_index(mem_stride, order[2]), dim4_index(mem_stride, order[3])};
  dim4 shape_permute = {
      dim4_index(mem_shape, order[0]), dim4_index(mem_shape, order[1]),
      dim4_index(mem_shape, order[2]), dim4_index(mem_shape, order[3])};
  return make_gtensor<dtype>(shape_permute, mode, addr, stride_permute);
}

template <typename dtype>
gtensor<dtype> &make_gtensor_permute(dim4 &mem_shape, tensor_mode_t mode,
                                     dtype *addr) {
  int default_order[4] = {0, 1, 2, 3};
  return make_gtensor_permute(mem_shape, mode, addr, default_order);
}

template <typename DataType> int get_data_size() {
  if constexpr (std::is_same_v<DataType, fp32> ||
                std::is_same_v<DataType, int32> ||
                std::is_same_v<DataType, uint32>) {
    return 4;
  } else if constexpr (std::is_same_v<DataType, fp16> ||
                       std::is_same_v<DataType, bf16> ||
                       std::is_same_v<DataType, int16> ||
                       std::is_same_v<DataType, uint16>) {
    return 2;
  } else if constexpr (std::is_same_v<DataType, int8> ||
                       std::is_same_v<DataType, uint8> ||
                       std::is_same_v<DataType, fp8e4m3> ||
                       std::is_same_v<DataType, fp8e5m2>) {
    return 1;
  } else {
    static_assert(false, "unsupported data type");
  }
}

template <typename DataType> int get_nic() {
  if constexpr (std::is_same_v<DataType, fp32>) {
    return 1;
  } else if constexpr (std::is_same_v<DataType, int4> ||
                       std::is_same_v<DataType, uint4> ||
                       std::is_same_v<DataType, fp4>) {
    return LANE_NUM * 2;
  } else {
#if defined(__sg2262__)
    if constexpr (std::is_same_v<DataType, int8> ||
                  std::is_same_v<DataType, uint8> ||
                  std::is_same_v<DataType, fp16> ||
                  std::is_same_v<DataType, bf16>) {
      return 16;
    } else if constexpr (std::is_same_v<DataType, fp8e4m3> ||
                         std::is_same_v<DataType, fp8e5m2>) {
      return 32;
    } else {
      static_assert(false, "unsupported data type");
    }
#elif defined(__bm1684x2__)
    if constexpr (std::is_same_v<DataType, int8> ||
                  std::is_same_v<DataType, uint8> ||
                  std::is_same_v<DataType, fp8e4m3> ||
                  std::is_same_v<DataType, fp8e5m2>) {
      return 32;
    } else if constexpr (std::is_same_v<DataType, fp16> ||
                         std::is_same_v<DataType, bf16>) {
      return 16;
    } else {
      static_assert(false, "unsupported data type");
    }
#elif defined(__mars3__)
    if constexpr (std::is_same_v<DataType, int8> ||
                  std::is_same_v<DataType, uint8>) {
      return 16;
    } else {
      return 8;
    }
#else
    return LANE_NUM / get_data_size<DataType>();
#endif
  }
}

/************************************************************************************
 */
/************************************************************************************
 */
/*        deprecated */
/**************************************************************************************/
/**************************************************************************************/

template <typename T>
void get_stride(dim4 *stride, dim4 *shape, align_mode_t mode,
                int start_idx = 0) {
  int eu_num = get_eu_num<T>();
  int stride_n = 0;
  int stride_c = 0;
  int stride_h = 0;
  int stride_w = 0;
  if (mode == TPU_ALIGN) {
    stride_h = shape->w;
    stride_c = shape->h * stride_h;
    stride_c = align(stride_c, eu_num);
    stride_n = div_up(start_idx + shape->c, LANE_NUM) * stride_c;
    stride_w = 1;

  } else if (mode == TPU_COMPACT) {
    stride_h = shape->w;
    stride_c = shape->h * stride_h;
    stride_n = div_up(start_idx + shape->c, LANE_NUM) * stride_c;
    stride_w = 1;

  } else if (mode == TPU_ROW_ALIGN) {
    stride_n = div_up(start_idx + shape->c, LANE_NUM) * stride_c;
    stride_c = shape->h * stride_h;
    stride_h = div_up(shape->w, eu_num);
    stride_w = 1;

  } else {
    stride_n = shape->c * shape->h * shape->w;
    stride_c = shape->h * shape->w;
    stride_h = shape->w;
    stride_w = 1;
  }
  *stride = {stride_n, stride_c, stride_h, stride_w};
}

template <typename U, typename V>
void aligned_stride_4d(dim4 *aligned_stride, dim4 *shape, U start_idx,
                       V eu_size) {
  int eu_num = EU_BYTES / eu_size;
  aligned_stride->h = shape->w;
  aligned_stride->c = shape->h * aligned_stride->h;
  aligned_stride->c = align(aligned_stride->c, eu_num);
  aligned_stride->n =
      div_up(start_idx + shape->c, LANE_NUM) * aligned_stride->c;
  aligned_stride->w = 1;
}

} // namespace ppl

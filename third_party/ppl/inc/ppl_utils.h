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

void swap(int &a, int &b) {
  int tmp = a;
  a = b;
  b = tmp;
}

void get_stride(dim4 *stride, dim4 *shape, align_mode_t mode, int eu_num = 0,
                int start_idx = 0) {
  if (mode == TPU_ALIGN) {
    stride->h = shape->w;
    stride->c = shape->h * stride->h;
    stride->c = align(stride->c, eu_num);
    stride->n = div_up(start_idx + shape->c, NPU_NUM) * stride->c;
    stride->w = 1;

  } else if (mode == TPU_COMPACT) {
    stride->h = shape->w;
    stride->c = shape->h * stride->h;
    stride->n = div_up(start_idx + shape->c, NPU_NUM) * stride->c;
    stride->w = 1;

  } else if (mode == TPU_ROW_ALIGN) {
    stride->n = div_up(start_idx + shape->c, NPU_NUM) * stride->c;
    stride->c = shape->h * stride->h;
    stride->h = div_up(shape->w, eu_num);
    stride->w = 1;

  } else {
    stride->n = shape->c * shape->h * shape->w;
    stride->c = shape->h * shape->w;
    stride->h = shape->w;
    stride->w = 1;
  }
}

template <typename U, typename V>
void aligned_stride_4d(dim4 *aligned_stride, dim4 *shape, U start_idx,
                       V eu_size) {
  get_stride(aligned_stride, shape, TPU_ALIGN, EU_BYTES / eu_size, start_idx);
}

template<typename T>
data_type_t convert_dtype() {
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
  } else {
    return DT_NONE;
  }
}

int dim4_index(const dim4* d, int index) {
  if (index < 0 && index >= -4)
    index = index + 4;
  if (index == 0) return d->n;
  else if (index == 1) return d->c;
  else if (index == 2) return d->h;
  else if (index == 3) return d->w;
  else return -1;
}

template <typename dtype, typename dimT>
tensor<dtype> &make_tensor(dimT &block_shape, dimT &real_shape,
                           align_mode_t align_mode = TPU_ALIGN) {
  tensor<dtype> new_tensor(block_shape, align_mode);
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
  dim4 stride_permute;
  dim4 shape_permute;
  dim4 mem_stride;
  mem_stride.n = mem_shape.c * mem_shape.h * mem_shape.w;
  mem_stride.c = mem_shape.h * mem_shape.w;
  mem_stride.h = mem_shape.w;
  mem_stride.w = 1;
  stride_permute.n = dim4_index(&mem_stride, order[0]);
  stride_permute.c = dim4_index(&mem_stride, order[1]);
  stride_permute.h = dim4_index(&mem_stride, order[2]);
  stride_permute.w = dim4_index(&mem_stride, order[3]);
  shape_permute.n = dim4_index(&mem_shape, order[0]);
  shape_permute.c = dim4_index(&mem_shape, order[1]);
  shape_permute.h = dim4_index(&mem_shape, order[2]);
  shape_permute.w = dim4_index(&mem_shape, order[3]);
  return make_gtensor<dtype>(shape_permute, mode, addr, stride_permute);
}

template <typename dtype>
gtensor<dtype> &make_gtensor_permute(dim4 &mem_shape, tensor_mode_t mode, dtype *addr) {
  int default_order[4] = {0, 1, 2, 3};
  return make_gtensor_permute(mem_shape, mode, addr, default_order);
}

template <typename DataType> int get_eu_num() {
  if constexpr (std::is_same_v<DataType, int4>) {
    return 2 * EU_BYTES;
  } else {
    return EU_BYTES / sizeof(DataType);
  }
}

template <typename DataType> int get_nic() {
  if constexpr (std::is_same_v<DataType, fp32>) {
    return 1;
  } else if constexpr (std::is_same_v<DataType, int4> ||
                       std::is_same_v<DataType, uint4>) {
    return LANE_NUM * 2;
  } else {
    return LANE_NUM / sizeof(DataType);
  }
}

} // namespace ppl

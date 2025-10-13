//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
//===----------------------------------------------------------------------===//


#pragma once
#include "ppl_defs.h"

namespace ppl {

enum comparision_mode_t {
  GREATER = 0,
  LESS = 1,
  EQUAL = 2,
};

enum rounding_mode_t {
  RM_HALF_TO_EVEN = 0,
  RM_HALF_AWAY_FROM_ZERO = 1,
  RM_TOWARDS_ZERO = 2,
  RM_DOWN = 3,
  RM_UP = 4,
  RM_HALF_UP = 5,
  RM_HALF_DOWN = 6,
};

enum coeff_table_mode_t {
  EXP = 0,
  LOG = 1,
  SIN = 2,
  COS = 3,
  TAN = 4,
  ARCSIN = 5,
  ERF_TAYLOR = 6,
  SERIAL_NUMBER = 30, // int32 uint32
};

enum transpose_mode_t {
  NC_TRANS = 0,
  CW_TRANS = 1,
};

enum align_mode_t {
  CONTINUOUS,
  TPU_ALIGN,
  TPU_COMPACT,
  TPU_ROW_ALIGN,
  NONE_ALIGN,
};

enum tensor_mode_t {
  L2 = 1,
  GLOBAL = 2,
};

enum data_type_t {
  DT_NONE = 0,
  DT_FP32,
  DT_FP16,
  DT_BF16,
  DT_FP8E5M2,
  DT_FP8E4M3,
  DT_FP20,
  DT_TF32,
  DT_INT32,
  DT_UINT32,
  DT_INT16,
  DT_UINT16,
  DT_INT8,
  DT_UINT8,
  DT_INT4,
  DT_UINT4,
  DT_INT64,
  DT_UINT64,
  DT_FP4,
};

enum all_reduce_opcode_t {
  ALL_REDUCE_NOP = 0,
  ALL_REDUCE_MUL = 1,
  ALL_REDUCE_MAX = 2,
  ALL_REDUCE_MIN = 3,
  ALL_REDUCE_ADD = 4,
  ALL_REDUCE_AMAX = 5,
  ALL_REDUCE_SUM = 6,
  ALL_REDUCE_POWERSUM = 7,
};

enum all_reduce_psum_t {
  ALL_REDUCE_PSUM_WO = 0,
  ALL_REDUCE_PSUM_WR = 1,
};

enum memory_alloc_method_t {
  BANK_CONFLICT_FIRST = 0,
  MEMORY_UTILIZATION_FIRST = 1,
};

enum sync_type {
  TIU_DMA_SYNC = 0x1,
  SDMA_SYNC = 0x10,
  HAU_SYNC = 0x100
};

#define SYNC_ALL TIU_DMA_SYNC & SDMA_SYNC & HAU_SYNC

struct dim4 {
  int n, c, h, w;
  dim4();
  dim4(const dim4 &);
  dim4 &operator=(const dim4 &);
  // dim4 &operator=(const dim4 &other) {
  //   n = other.n;
  //   c = other.c;
  //   h = other.h;
  //   w = other.w;
  //   return *this;
  // };
  dim4(int _n, int _c, int _h, int _w);
};

struct dim2 {
  int h, w;
  dim2();
  // dim2(const dim4 &);
  dim2 &operator=(const dim4 &);
  dim2(int _h, int _w);
};

struct padding_t {
  int top, bottom, left, right;
  padding_t();
  padding_t(const padding_t &);
  padding_t &operator=(const padding_t &);
  padding_t(int _top, int _bottom, int _left, int _right);
};

template <typename dtype> struct tensor {
public:
  tensor(align_mode_t align_mode = TPU_ALIGN, long long address = -1);
  // tensor(const tensor&) = delete;
  tensor &operator=(const tensor &) = delete;

  template <typename dimT>
  explicit tensor(dimT &_shape, align_mode_t align_mode = TPU_ALIGN,
                  long long address = -1);
  template <typename dtype2> tensor<dtype2> &view();
  tensor<dtype> &view(align_mode_t align_mode);
  template <typename dtype2, typename dimT> tensor<dtype2> &view(dimT &_shape);
  template <typename dtype2, typename dimT>
  tensor<dtype2> &view(dimT &_shape, dimT &_stride);
  template <typename dimT> tensor<dtype> &view(dimT &_shape);
  template <typename dimT> tensor<dtype> &view(dimT &_shape, dimT &_stride);
  template <typename dimT> tensor<dtype> &sub_view(dimT &_shape, dimT &_offset);

  dim4& getBlockShape();
  dim4& getShape();
private:
  dtype *data;
};

template <typename dtype> struct gtensor {
public:
  gtensor(tensor_mode_t mode, dtype *address = nullptr);
  // gtensor(const gtensor&) = delete;
  gtensor &operator=(const gtensor &) = delete;

  template <typename dimT>
  explicit gtensor(dimT &_shape, tensor_mode_t mode, dtype *address = nullptr);
  template <typename dtype2> gtensor<dtype2> &view();

  template <typename dtype2, typename dimT> gtensor<dtype2> &view(dimT &_shape);
  template <typename dtype2, typename dimT>
  gtensor<dtype2> &view(dimT &_shape, dimT &_stride);
  template <typename dimT> gtensor<dtype> &view(dimT &_shape);
  template <typename dimT> gtensor<dtype> &view(dimT &_shape, dimT &_stride);
  template <typename dimT>
  gtensor<dtype> &sub_view(dimT &_shape, dimT &_offset);

private:
  dtype *data;
};

} // namespace ppl

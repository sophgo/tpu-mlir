#pragma once
#include "ppl_types.h"
#include "ppl_utils.h"

namespace ppl {
/*
 * Note:
 * 1. The random number generation interval is [min_val, max_val)
 */
template <typename DataType0, typename DataType1>
void rand(DataType0 *ptr, dim4 *shape, dim4 *stride, dim4 *offset,
          DataType1 min_val, DataType1 max_val);

template <typename DataType0, typename DataType1>
void rand(DataType0 *ptr, dim4 *shape, dim4 *stride, DataType1 min_val,
          DataType1 max_val) {
  rand(ptr, shape, stride, (dim4 *)nullptr, min_val, max_val);
}

template <typename DataType0, typename DataType1>
void rand(DataType0 *ptr, dim4 *shape, DataType1 min_val, DataType1 max_val) {
  rand(ptr, shape, (dim4 *)nullptr, (dim4 *)nullptr, min_val, max_val);
}

// malloc and rand with result
template <typename DataType0, typename DataType1>
DataType0 *rand(dim4 *shape, dim4 *stride, dim4 *offset, DataType1 min_val,
                DataType1 max_val);

template <typename DataType0, typename DataType1>
DataType0 *rand(dim4 *shape, dim4 *stride, DataType1 min_val,
                DataType1 max_val) {
  return rand<DataType0>(shape, stride, (dim4 *)nullptr, min_val, max_val);
}

template <typename DataType0, typename DataType1>
DataType0 *rand(dim4 *shape, DataType1 min_val, DataType1 max_val) {
  return rand<DataType0>(shape, (dim4 *)nullptr, (dim4 *)nullptr, min_val,
                         max_val);
}

template <typename DataType> DataType *rand(dim4 *shape) {
  return rand<DataType>(shape, (dim4 *)nullptr, (dim4 *)nullptr, 0, 0);
}

template <typename DataType> DataType *malloc(dim4 *shape);

void enable_pipeline();

void set_core_num(int num);

int get_core_num();

int get_core_index();

void tpu_sync_core();

void sync();

template <typename DataType> void assert(DataType condition);

template <typename DataType>
void read_npy(DataType *dst, const char *file_path);

template <typename DataType>
void read_npz(DataType *dst, const char *file_path, const char *tensor_name);

template <typename DstType, typename FileType>
void read_bin(DstType *dst, const char *file_path, FileType file_dtype);

template <typename DstType, typename FileType>
void read_bin(DstType *dst, const char *file_path) {
  read_bin(dst, file_path, (FileType)0);
}

void hau_poll();

void tpu_poll();

template <typename DataType> int64 get_gmem_addr(gtensor<DataType> &src);

// template <typename DataType>
// uint64 get_gmem_addr(DataType *address);

template <typename DataType> DataType get_value(int64 gaddr);

void msg_send(int msg_idx, int wait_cnt, bool is_dma);

void msg_wait(int msg_idx, int send_cnt, bool is_dma);

void fence();

void lane_mask(int mask, bool long_valid);

void vset(int ew, int lmul, int v_len);

template <typename... Args> void print(const char *format, Args... args);

template <typename DataType> char *to_string(tensor<DataType> &src);

template <typename DataType> char *to_string(gtensor<DataType> &src);

} // namespace ppl

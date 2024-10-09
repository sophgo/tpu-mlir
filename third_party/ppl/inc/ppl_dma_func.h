#pragma once
#include "ppl_defs.h"
#include "ppl_tpu.h"
#include "ppl_types.h"

namespace ppl {
namespace dma {

template <typename DataType>
void load(tensor<DataType> &dst, gtensor<DataType> &src);
template <typename DataType>
void load_compact(tensor<DataType> &dst, gtensor<DataType> &src);

template <typename DataType>
void store(gtensor<DataType> &dst, tensor<DataType> &src);

template <typename DataType>
void load_transpose_cw(tensor<DataType> &dst, gtensor<DataType> &src);
template <typename DataType>
void load_transpose_nc(tensor<DataType> &dst, gtensor<DataType> &src);

template <typename DataType>
void load_transpose(tensor<DataType> &dst, gtensor<DataType> &src,
                    transpose_mode_t trans_mode) {
  if (trans_mode == CW_TRANS) {
    return load_transpose_cw(dst, src);
  } else {
    return load_transpose_nc(dst, src);
  }
}

template <typename DataType>
void store_transpose_cw(gtensor<DataType> &dst, tensor<DataType> &src);
template <typename DataType>
void store_transpose_nc(gtensor<DataType> &dst, tensor<DataType> &src);

template <typename DataType>
void store_transpose(gtensor<DataType> &dst, tensor<DataType> &src,
                     transpose_mode_t trans_mode) {
  if (trans_mode == CW_TRANS) {
    return store_transpose_cw(dst, src);
  } else {
    return store_transpose_nc(dst, src);
  }
}

template <typename DataType>
void load_broadcast(tensor<DataType> &dst, gtensor<DataType> &src,
                    int num = LANE_NUM);

template <typename DataType>
void broadcast(tensor<DataType> &dst, tensor<DataType> &src,
               int num = LANE_NUM);

template <typename DataType>
void move(tensor<DataType> &dst, tensor<DataType> &src);

template <typename DataType>
void move(gtensor<DataType> &dst, gtensor<DataType> &src);

template <typename DataType>
void transpose_cw(gtensor<DataType> &dst, gtensor<DataType> &src);
template <typename DataType>
void transpose_nc(gtensor<DataType> &dst, gtensor<DataType> &src);
template <typename DataType>
void transpose(gtensor<DataType> &dst, gtensor<DataType> &src,
               transpose_mode_t trans_mode) {
  if (trans_mode == CW_TRANS) {
    return transpose_cw(dst, src);
  } else {
    return transpose_nc(dst, src);
  }
}

template <typename DataType>
void transpose_cw(tensor<DataType> &dst, tensor<DataType> &src);
template <typename DataType>
void transpose_nc(tensor<DataType> &dst, tensor<DataType> &src);
template <typename DataType>
void transpose(tensor<DataType> &dst, tensor<DataType> &src,
               transpose_mode_t trans_mode) {
  if (trans_mode == CW_TRANS) {
    return transpose_cw(dst, src);
  } else {
    return transpose_nc(dst, src);
  }
}

template <typename DataType0, typename DataType1>
void fill(gtensor<DataType0> &dst, DataType1 C);
template <typename DataType0, typename DataType1>
void fill(tensor<DataType0> &dst, DataType1 C);

template <typename DataType0> void zero(tensor<DataType0> &dst) {
  fill(dst, 0);
}
template <typename DataType0> void zero(gtensor<DataType0> &dst) {
  fill(dst, 0);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(tensor<DataType0> &dst, tensor<DataType1> &param,
              tensor<DataType2> &index, DataType3 C, int index_start_pos);
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(tensor<DataType0> &dst, tensor<DataType1> &param,
              tensor<DataType2> &index, DataType3 C) {
  gather_h(dst, param, index, C, 0);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(tensor<DataType0> &dst, gtensor<DataType1> &param,
              tensor<DataType2> &index, DataType3 C, int index_start_pos);
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(tensor<DataType0> &dst, gtensor<DataType1> &param,
              tensor<DataType2> &index, DataType3 C) {
  gather_h(dst, param, index, C, 0);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(tensor<DataType0> &dst, tensor<DataType1> &param,
              gtensor<DataType2> &index, DataType3 C, int index_start_pos);
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(tensor<DataType0> &dst, tensor<DataType1> &param,
              gtensor<DataType2> &index, DataType3 C) {
  gather_h(dst, param, index, C, 0);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(tensor<DataType0> &dst, gtensor<DataType1> &param,
              gtensor<DataType2> &index, DataType3 C, int index_start_pos);
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(tensor<DataType0> &dst, gtensor<DataType1> &param,
              gtensor<DataType2> &index, DataType3 C) {
  gather_h(dst, param, index, C, 0);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(gtensor<DataType0> &dst, tensor<DataType1> &param,
              tensor<DataType2> &index, DataType3 C, int index_start_pos);
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(gtensor<DataType0> &dst, tensor<DataType1> &param,
              tensor<DataType2> &index, DataType3 C) {
  gather_h(dst, param, index, C, 0);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(gtensor<DataType0> &dst, gtensor<DataType1> &param,
              tensor<DataType2> &index, DataType3 C, int index_start_pos);
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(gtensor<DataType0> &dst, gtensor<DataType1> &param,
              tensor<DataType2> &index, DataType3 C) {
  gather_h(dst, param, index, C, 0);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(gtensor<DataType0> &dst, tensor<DataType1> &param,
              gtensor<DataType2> &index, DataType3 C, int index_start_pos);
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(gtensor<DataType0> &dst, tensor<DataType1> &param,
              gtensor<DataType2> &index, DataType3 C) {
  gather_h(dst, param, index, C, 0);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(gtensor<DataType0> &dst, gtensor<DataType1> &param,
              gtensor<DataType2> &index, DataType3 C, int index_start_pos);
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(gtensor<DataType0> &dst, gtensor<DataType1> &param,
              gtensor<DataType2> &index, DataType3 C) {
  gather_h(dst, param, index, C, 0);
}

template <typename DataType0, typename DataType1, typename DataType2>
void scatter_h(tensor<DataType0> &dst, tensor<DataType1> &param,
               tensor<DataType2> &index);

template <typename DataType0, typename DataType1, typename DataType2>
void scatter_h(tensor<DataType0> &dst, gtensor<DataType1> &param,
               tensor<DataType2> &index);

template <typename DataType0, typename DataType1, typename DataType2>
void scatter_h(tensor<DataType0> &dst, tensor<DataType1> &param,
               gtensor<DataType2> &index);

template <typename DataType0, typename DataType1, typename DataType2>
void scatter_h(tensor<DataType0> &dst, gtensor<DataType1> &param,
               gtensor<DataType2> &index);

template <typename DataType0, typename DataType1, typename DataType2>
void scatter_h(gtensor<DataType0> &dst, tensor<DataType1> &param,
               tensor<DataType2> &index);

template <typename DataType0, typename DataType1, typename DataType2>
void scatter_h(gtensor<DataType0> &dst, gtensor<DataType1> &param,
               tensor<DataType2> &index);

template <typename DataType0, typename DataType1, typename DataType2>
void scatter_h(gtensor<DataType0> &dst, tensor<DataType1> &param,
               gtensor<DataType2> &index);

template <typename DataType0, typename DataType1, typename DataType2>
void scatter_h(gtensor<DataType0> &dst, gtensor<DataType1> &param,
               gtensor<DataType2> &index);

template <typename DataType0, typename DataType1>
uint nonzero(gtensor<DataType0> &dst, tensor<DataType1> &src);

template <typename DataType0, typename DataType1>
uint nonzero(gtensor<DataType0> &dst, gtensor<DataType1> &src);

template <typename DataType>
void move_cross_lane(tensor<DataType> &dst, tensor<DataType> &src);

template <typename DataType0, typename DataType1>
unsigned int mask_select(gtensor<DataType0> &dst, tensor<DataType0> &src, tensor<DataType1> &mask);

template <typename DataType0, typename DataType1>
unsigned int mask_select(gtensor<DataType0> &dst, tensor<DataType0> &src, gtensor<DataType1> &mask);

template <typename DataType0, typename DataType1>
unsigned int mask_select(gtensor<DataType0> &dst, gtensor<DataType0> &src, tensor<DataType1> &mask);

template <typename DataType0, typename DataType1>
unsigned int mask_select(gtensor<DataType0> &dst, gtensor<DataType0> &src, gtensor<DataType1> &mask);

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void mask_batch_bcast(tensor<DataType0> &dst, tensor<DataType1> &count,
                      tensor<DataType2> &src, tensor<DataType3> &mask,
                      bool is_repeat);

template <typename DataType>
void reverse(tensor<DataType> &dst, tensor<DataType> &src, int dim);
template <typename DataType>
void reverse(tensor<DataType> &dst, gtensor<DataType> &src, int dim);
template <typename DataType>
void reverse(gtensor<DataType> &dst, tensor<DataType> &src, int dim);
template <typename DataType>
void reverse(gtensor<DataType> &dst, gtensor<DataType> &src, int dim);

template <typename DataType>
void vload(int dst_v_idx, gtensor<DataType> &src);

template <typename DataType>
void vstore(gtensor<DataType> &dst, int v_idx);

template <typename DataType>
void move_tv(tensor<DataType> &dst, int v_idx);
void move_tv(int smem_offset, int v_idx);

template <typename DataType>
void move_distv(tensor<DataType> &dst, int v_idx);

template <typename DataType>
void move_vv(tensor<DataType> &dst, int v_idx0, int v_idx1);
void move_vv(int smem_offset, int v_idx0, int v_idx1);

template <typename DataType>
void move_distvv(tensor<DataType> &dst, int v_idx0, int v_idx1);

template <typename DataType>
void move_vt(int dst_v_idx, tensor<DataType> &src);
void move_vt(int dst_v_idx, int smem_offset);

template <typename DataType>
void move_vcoll(int dst_v_idx, tensor<DataType> &src);

} // namespace dma
} // namespace ppl

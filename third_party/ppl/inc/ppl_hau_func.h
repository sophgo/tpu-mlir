#pragma once
#include "ppl_defs.h"
#include "ppl_types.h"

namespace ppl {
namespace hau {

// tpu_hau_sort_specific_index(bm1684x and bm1690)
// tpu_hau_sort_specific_index_2d(bm1690)
template <typename DataType>
void topk(gtensor<DataType> &dst, gtensor<int> &dst_idx,
          gtensor<DataType> &src, gtensor<int> &src_idx,
          int K, bool descended);

// tpu_hau_sort(bm1684x and bm1690)
// tpu_hau_sort_2d(bm1690)
template <typename DataType>
void topk(gtensor<DataType> &dst, gtensor<DataType> &src,
          int K, bool descended) {
  gtensor<int> *dst_idx = nullptr;
  gtensor<int> *src_idx = nullptr;
  topk(dst, *dst_idx, src, *src_idx, K, descended);
}

// tpu_hau_sort_natural_index(bm1684x and bm1690)
// tpu_hau_sort_natural_index_2d(bm1690)
template <typename DataType>
void topk(gtensor<DataType> &dst, gtensor<int> &dst_idx,
          gtensor<DataType> &src, int K, bool descended) {
  gtensor<int> *src_idx = nullptr;
  topk(dst, dst_idx, src, *src_idx, K, descended);
}

// tpu_hau_gather_line only in bm1684x
template <typename DataType0, typename DataType1, typename DataType2>
void gather_line(gtensor<DataType0> &dst, gtensor<DataType0> &param,
                 gtensor<DataType1> &index, DataType2 C, int start,
                 int end, bool fill_const);

} // namespace hau
} // namespace ppl

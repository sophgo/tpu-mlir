#pragma once
#include "ppl_defs.h"
#include "ppl_types.h"

namespace ppl {
namespace sdma {

template <typename DataType>
void move(gtensor<DataType> &dst, gtensor<DataType> &src, int port_id);

template <typename DataType>
void move(gtensor<DataType> &dst, gtensor<DataType> &src) {
  return move(dst, src, DEFAULT_SDMA_PORT);
}

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
} // namespace sdma
} // namespace ppl

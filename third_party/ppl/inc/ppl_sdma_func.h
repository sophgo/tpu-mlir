//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "ppl_defs.h"
#include "ppl_types.h"

namespace ppl {
namespace sdma {

template <typename DataType0, typename DataType1>
void fill(gtensor<DataType0> &dst, DataType1 C,
          int port_id = DEFAULT_SDMA_PORT);

template <typename DataType0>
void zero(gtensor<DataType0> &dst, int port_id = DEFAULT_SDMA_PORT) {
  fill(dst, 0, port_id);
}

template <typename DataType>
void move(gtensor<DataType> &dst, gtensor<DataType> &src,
          int port_id = DEFAULT_SDMA_PORT);

template <typename DataType>
void transpose_cw(gtensor<DataType> &dst, gtensor<DataType> &src,
                  int port_id = DEFAULT_SDMA_PORT);

template <typename DataType>
void transpose_nc(gtensor<DataType> &dst, gtensor<DataType> &src,
                  int port_id = DEFAULT_SDMA_PORT);

template <typename DataType>
void transpose(gtensor<DataType> &dst, gtensor<DataType> &src,
               transpose_mode_t trans_mode, int port_id = DEFAULT_SDMA_PORT) {
  if (trans_mode == CW_TRANS) {
    return transpose_cw(dst, src, port_id);
  } else {
    return transpose_nc(dst, src, port_id);
  }
}

template <typename DataType1, typename DataType2>
void reduce(gtensor<DataType1> &dst, DataType2 &src, all_reduce_psum_t psum,
            all_reduce_opcode_t opcode);

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gather_h(gtensor<DataType0> &dst, gtensor<DataType1> &param,
              gtensor<DataType2> &index, DataType3 C,
              int port_id = DEFAULT_SDMA_PORT);

template <typename DataType0, typename DataType1, typename DataType2>
void scatter_h(gtensor<DataType0> &dst, gtensor<DataType1> &param,
               gtensor<DataType2> &index, int port_id = DEFAULT_SDMA_PORT);

} // namespace sdma
} // namespace ppl

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace tpu_mlir {

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  int64_t out_addr;
  int64_t out_size;
  int64_t buffer_addr;
  int64_t buffer_size;
  int64_t n_idx;
  int64_t n_slice;
  int64_t c_idx;
  int64_t c_slice;
  int64_t h_idx;
  int64_t h_slice;
  int64_t d_idx;
  int64_t d_slice;
  int64_t w_idx;
  int64_t w_slice;
  int64_t h_idx_offset;
  int64_t id;
  int64_t stage;
  int64_t type;
  bool eu_align;
  bool overstepped;
} group_info_t;

typedef struct local_sec_info {
  int32_t group_type;

  int32_t n_slice;
  int32_t out_n_slice;

  int32_t d_slice;
  // int32_t out_d_slice; // <- if add this, need to change along with backend api_common.h:sec_info, otherwise memcpy will mess up

  int32_t is_h_split;
  int32_t h_idx;
  int32_t h_slice;
  int32_t out_h_idx;
  int32_t out_h_slice;

  int32_t is_w_split;
  int32_t w_idx;
  int32_t w_slice;
  int32_t out_w_idx;
  int32_t out_w_slice;

  int32_t is_c_split;
  int32_t c_idx;
  int32_t c_slice;

  int32_t n_idx;
  int32_t d_idx;

  void print() {
    printf("group_type:%d, n_idx: %d, n_slice:%d, out_n_slice:%d, d_idx:%d, d_slice:%d, \
            >>>>>>>>is_h_split:%d, h_idx:%d, h_slice:%d, out_h_idx:%d, out_h_slice:%d, \
            >>>>>>>>is_w_split:%d, w_idx:%d, w_slice:%d, out_w_idx:%d, out_w_slice:%d, \
            >>>>>>>>is_c_split:%d, c_idx:%d, c_slice:%d\n",
            group_type, n_idx, n_slice, out_n_slice, d_idx, d_slice, \
            is_h_split, h_idx, h_slice, out_h_idx, out_h_slice, is_w_split, \
            w_idx, w_slice, out_w_idx, out_w_slice, is_c_split, c_idx, c_slice);
  }
} local_sec_info_t;

#ifdef __cplusplus
}
#endif

mlir::LogicalResult BroadCastBinaryLocalGenSupport(mlir::Operation *op);

} // namespace tpu_mlir

#include "tpu_mlir/Support/Module.h"

/// Include the ODS generated interface header files.
#include "tpu_mlir/Interfaces/LocalGenInterface.h.inc"
#include "tpu_mlir/Interfaces/DynLocalGenInterface.h.inc"

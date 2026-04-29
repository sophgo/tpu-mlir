//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "concat_slice.h"
#include "helper.h"
#include "ppl_static_host.h"
#include "tpu_mlir/Backend/BM168x/Param.h"
#include <assert.h>
#include <cstdio>
#include <functional>
#include <stddef.h>
#include <stdint.h>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif
// ======================================
// Global GenInterface
// ======================================

void concat_slice_tiling(gaddr_t ptr_out, gaddr_t ptr_in0, gaddr_t ptr_in1,
                         int outer_size, int axis_size_0, int axis_size_1,
                         int inner_size, int &c, int &h, int &block_h,
                         int core_num, int32_t dtype) {

  if (inner_size != 1) {
    printf("Error: concat_slice only support inner_size=1 now, but got %d\n",
           inner_size);
    exit(-1);
  }
  int npu_num, npu_size;
  get_chip_info(&npu_num, &npu_size);
  if (outer_size % npu_num != 0) {
    printf("Error: concat_slice outer_size should be divisible by npu_num, but "
           "got outer_size=%d, npu_num=%d\n",
           outer_size, npu_num);
    exit(-1);
  }
  c = npu_num;
  h = outer_size / c;
  auto func = dtype == DTYPE_BFP16 ? concat_slice_bf16 : concat_slice_f16;
  block_h = h / core_num;
  int ret = 0;
  while (block_h > 0) {
    printf("ConcatSlice try block_h:%d/%d\n", block_h, h);
    ret = func(ptr_out, ptr_in0, ptr_in1, c, h, axis_size_0, axis_size_1,
               block_h, core_num);
    if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr) {
      block_h = block_h / 2;
    } else {
      break;
    }
  }
  if (ret != 0 || block_h == 0) {
    printf("Error: concat_slice kernel returned %d\n", ret);
    exit(-1);
  }
}

// static interface
void api_concat_slice_global(void *param, size_t param_size, void *input,
                             void *output) {
  auto *_param = (concat_slice_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input;
  tensor_spec_t *out_spec = (tensor_spec_t *)output;
  auto func =
      in_spec[0].dtype == DTYPE_BFP16 ? concat_slice_bf16 : concat_slice_f16;
  const int core_num = get_core_num();
  int axis = _param->axis;
  int dims = in_spec[0].dims;

  int outer_size = 1;
  for (int i = 0; i < axis; i++)
    outer_size *= in_spec[0].shape[i];

  int axis_size_0 = in_spec[0].shape[axis];
  int axis_size_1 = in_spec[1].shape[axis];

  int inner_size = 1;
  for (int i = axis + 1; i < dims; i++) {
    inner_size *= in_spec[0].shape[i];
  }
  int block_h, c, h;
  concat_slice_tiling(out_spec[0].addr, in_spec[0].addr, in_spec[1].addr,
                      outer_size, axis_size_0, axis_size_1, inner_size, c, h,
                      block_h, core_num, in_spec[0].dtype);
}

// dynamic interface
int api_dyn_concat_slice_global(void *param, void *input, void *output,
                                void *buffer) {
  auto *_param = (concat_slice_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input;
  tensor_spec_t *out_spec = (tensor_spec_t *)output;
  const int core_num = get_core_num();
  int axis = _param->axis;
  int dims = in_spec[0].dims;

  int outer_size = 1;
  for (int i = 0; i < axis; i++)
    outer_size *= in_spec[0].shape[i];

  int axis_size_0 = in_spec[0].shape[axis];
  int axis_size_1 = in_spec[1].shape[axis];

  int inner_size = 1;
  for (int i = axis + 1; i < dims; i++)
    inner_size *= in_spec[0].shape[i];

  auto func = in_spec->dtype == DTYPE_BFP16 ? fill_concat_slice_bf16_struct
                                            : fill_concat_slice_f16_struct;
  int block_h = outer_size;
  int c = 1, h = 1;
  if (buffer != nullptr) {
    concat_slice_tiling(out_spec[0].addr, in_spec[0].addr, in_spec[1].addr,
                        outer_size, axis_size_0, axis_size_1, inner_size, c, h,
                        block_h, core_num, in_spec[0].dtype);
  }
  return func(out_spec[0].addr, in_spec[0].addr, in_spec[1].addr, c, h,
              axis_size_0, axis_size_1, block_h, core_num, buffer);
}

#ifdef __cplusplus
}
#endif

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "all_sorted_topk.h"
#include "ppl_static_host.h"
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

// static interface
void api_all_sorted_topk_global(void *param, size_t param_size, void *input,
                                void *output) {
  auto *_param = (topk_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input;
  tensor_spec_t *out_spec = (tensor_spec_t *)output;
  auto func = in_spec[0].dtype == DTYPE_FP32   ? allsort_topk_fp32
              : in_spec[0].dtype == DTYPE_FP16 ? allsort_topk_fp16
                                               : allsort_topk_bf16;
  const int core_num = get_core_num();
  int batch = 1;
  assert(_param->dim == in_spec[0].dims - 1);
  int select = in_spec[0].shape[in_spec[0].dims - 1];
  for (int i = 0; i < in_spec[0].dims - 1; i++) {
    batch *= in_spec[0].shape[i];
  }
  int ret = func(in_spec[0].addr, out_spec[0].addr, out_spec[1].addr,
                 _param->buffer_val_addr, _param->buffer_idx_addr,
                 _param->buffer_scatter_idx_addr, _param->seq_index_addr, batch,
                 select, select, _param->k, core_num);
  if (ret != 0) {
    printf("Error: all_sorted_topk kernel returned %d\n", ret);
    exit(-1);
  }
}

#ifdef __cplusplus
}
#endif

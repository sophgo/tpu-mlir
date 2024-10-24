#include "helper.h"
#include "softmax_h_dim.h"
#include "softmax_w_dim.h"

#include "tpu_mlir/Backend/BM168x/Param.h"

#include <assert.h>
#include <cstdio>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <tuple>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

void api_softmax_global(void *param, size_t param_size, void *input_spec,
                        void *output_spec, const char *chip, void *cmdid) {
  softmax_global_param_t *_param = (softmax_global_param_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  assert(in_spec->dtype == DTYPE_FP32 && out_spec->dtype == DTYPE_FP32);

  // tiling
  int begin_axis = _param->common.begin_axis;
  int end_axis = _param->common.end_axis;
  int outer = 1, chns = 1, iner = 1;
  for (int i = 0; i < begin_axis; i++) {
    outer *= in_spec->shape[i];
    assert(outer <= INT32_MAX);
  }
  for (int i = begin_axis; i <= end_axis; i++) {
    chns *= in_spec->shape[i];
    assert(chns <= INT32_MAX);
  }
  for (int i = end_axis + 1; i < in_spec->dims; i++) {
    iner *= in_spec->shape[i];
    assert(iner <= INT32_MAX);
  }

  int lane_num = 64;
  std::string chip_str(chip);
  if (chip_str == PPL_BM1688) {
    // TODO fix
    lane_num = 32;
  }
  int c_align = lane_num;
  int h_align = 1;
  int w_align = 1;

  int c_secs = 1;
  int h_secs = 1;
  int w_secs = 1;

  int n, c, h, w, block_c, block_h, block_w;
  block_c = align_up(outer / c_secs, c_align);
  n = 1;
  c = outer;
  if (iner == 1) {
    h = 1;
    w = chns;
    block_w = align_up(chns / w_secs, w_align);
  } else {
    h = chns;
    w = iner;
    block_w = align_up(iner / w_secs, w_align);
    block_h = align_up(chns / h_secs, h_align);
  }

  auto call_kernel = [&]() {
    if (iner == 1) {
      printf("------------------  w_dim  -------------------\n");
      return softmax_w_dim(chip, cmdid, out_spec->addr, in_spec->addr, n, c, h,
                           w, block_c, block_w);
    } else {
      printf("------------------  h_dim  -------------------\n");
      return softmax_h_dim(chip, cmdid, out_spec->addr, in_spec->addr, n, c, h,
                           w, block_c, block_h, block_w);
    }
  };

  auto split_func = [&](std::vector<std::tuple<int *, int, int>> &value) {
    for (auto &item : value) {
      int max_val = std::get<1>(item);
      // Make sure to loop at least twice so that you can run the pipeline in
      // parallel, secs = 2, todo
      for (int secs = 1; secs < max_val; ++secs) {
        int *block_ptr = std::get<0>(item);
        *block_ptr = align_up(max_val / secs, std::get<2>(item));
        int ret = call_kernel();
        CHECK_PPL_RET(ret);
        if (!ret) {
          return 0;
        }
      }
    }
    return 1;
  };

  std::vector<std::tuple<int *, int, int>> value;
  value.emplace_back(&block_c, outer, c_align);
  if (iner == 1) {
    // split order c->w
    value.emplace_back(&block_w, chns, w_align);
  } else {
    // split order c->w->h
    value.emplace_back(&block_w, iner, w_align);
    value.emplace_back(&block_h, chns, h_align);
  }

  int ret = split_func(value);
  printf("block [n:1, c:%d, h:%d, w:%d] vs total[n:1, c:%d, h:%d, w:%d]\n",
         block_c, block_h, block_w, c, h, w);

  if (ret != 0) {
    assert(0 && "tiling failed\n");
  }
}

#ifdef __cplusplus
}
#endif

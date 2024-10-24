#include "helper.h"
#include "rms_norm_bf16.h"
#include "rms_norm_fp16.h"
#include "rms_norm_fp32.h"

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

void api_rms_norm_global(void *param, size_t param_size, void *input_spec,
                         void *output_spec, const char *chip, void *cmdid) {
  rms_norm_global_spec_t *_param = (rms_norm_global_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  tensor_spec_t *weight_spec = nullptr;
  bool has_weight = (_param->common.affine & 0x1);
  float exp = _param->common.eps;
  if (has_weight) {
    weight_spec = in_spec + 1;
  }

  // tiling
  int outer = 1, chns = 1;
  for (int i = 0; i < in_spec->dims - 1; i++) {
    outer *= in_spec->shape[i];
    assert(outer <= INT32_MAX);
  }
  for (int i = in_spec->dims - 1; i < in_spec->dims; i++) {
    chns *= in_spec->shape[i];
    assert(chns <= INT32_MAX);
  }

  int w_align = 1;
  int w_secs = 1;
  int n = 1, c = outer, h = 1, w = chns;
  int block_w = align_up(chns / w_secs, w_align);

  auto call_kernel = [&]() {
    if (in_spec[0].dtype == DTYPE_FP16) {
      printf("------------------  DTYPE_FP16  -------------------\n");
      return rms_norm_fp16(chip, cmdid, out_spec->addr, in_spec->addr,
                           has_weight ? weight_spec->addr : 0, exp, has_weight,
                           n, c, h, w, block_w);
    } else if (in_spec[0].dtype == DTYPE_BFP16) {
      printf("------------------  DTYPE_BFP16  -------------------\n");
      return rms_norm_bf16(chip, cmdid, out_spec->addr, in_spec->addr,
                           has_weight ? weight_spec->addr : 0, exp, has_weight,
                           n, c, h, w, block_w);
    } else if (in_spec[0].dtype == DTYPE_FP32) {
      printf("------------------  DTYPE_FP32  -------------------\n");
      return rms_norm_fp32(chip, cmdid, out_spec->addr, in_spec->addr,
                           has_weight ? weight_spec->addr : 0, exp, has_weight,
                           n, c, h, w, block_w);
    } else {
      assert(0 && "unsupport data type!!!");
    }
  };

  auto split_func = [&](std::vector<std::tuple<int *, int, int>> &value) {
    for (auto &item : value) {
      int max_val = std::get<1>(item);
      for (int secs = 2; secs < max_val; ++secs) {
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
  // split order c->w
  value.emplace_back(&block_w, chns, w_align);

  int ret = split_func(value);
  printf("block [n:1, c:64, h:1, w:%d] vs total[n:1, c:%d, h:1, w:%d]\n",
         block_w, c, w);

  if (ret != 0) {
    assert(0 && "tiling failed\n");
  }
}

#ifdef __cplusplus
}
#endif

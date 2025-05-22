#include "correlation_fp.h"
#include "helper.h"
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

int call(int dtype, uint64_t ptr_dst, uint64_t ptr_lsrc, uint64_t ptr_rsrc,
         int N, int C, int H, int W, int max_disp) {
  int ret = -1;
  switch (dtype) {
  case DTYPE_FP32:
    ret = correlation_f32(ptr_dst, ptr_lsrc, ptr_rsrc, N, C, H, W, max_disp);
    CHECK_PPL_RET(ret);
    break;
  case DTYPE_FP16:
    ret = correlation_fp16(ptr_dst, ptr_lsrc, ptr_rsrc, N, C, H, W, max_disp);
    CHECK_PPL_RET(ret);
    break;
  case DTYPE_BFP16:
    ret = correlation_bf16(ptr_dst, ptr_lsrc, ptr_rsrc, N, C, H, W, max_disp);
    CHECK_PPL_RET(ret);
    break;
  default:
    assert(0 && "not supported, need fix correlation_fp.pl\n");
  }
  if (ret == 0) {
    return 0;
  } else {
    assert(0 && "not supported\n");
    return ret;
  }
  return 1;
}

void api_correlation_global(void *param, size_t param_size, void *input_spec,
                            void *output_spec, const int core_num,
                            const char *chip, void *cmdid) {

  correlation_global_param_t *_param = (correlation_global_param_t *)param;
  tensor_spec_t *in_l_spec = (tensor_spec_t *)input_spec;
  auto in_r_spec = in_l_spec + 1;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  auto max_disp = _param->max_disp;
  auto num_groups = _param->num_groups;
  int C = in_l_spec->shape[1] / num_groups;

  call(in_l_spec->dtype, out_spec->addr, in_l_spec->addr, in_r_spec->addr,
       num_groups, C, in_l_spec->shape[2], in_l_spec->shape[3], max_disp);
}

#ifdef __cplusplus
}
#endif

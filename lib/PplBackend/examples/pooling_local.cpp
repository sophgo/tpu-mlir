#include "avg_pool_2d_local.h"
#include "helper.h"
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

void api_avgpool_local(void *param, size_t param_size, void *slice_info,
                       void *input_spec, void *output_spec) {
  pooling_local_spec_t *_param = (pooling_local_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;

  int lane_num = 64;
  int eu_bytes = 64;
  std::string chip_str = get_chip_str();
  if (chip_str == PPL_BM1688) {
    lane_num = 32;
    eu_bytes = 32;
  }
  int c_align = lane_num;

  int output_addr = (int)out_spec->addr;
  int input_addr = (int)in_spec->addr;
  int kh = _param->common.kh;
  int kw = _param->common.kw;
  int stride_h = _param->common.stride_h;
  int stride_w = _param->common.stride_w;
  int pad_h_t = _param->common.pad_h_t;
  int pad_h_b = _param->common.pad_h_b;
  int pad_w_l = _param->common.pad_w_l;
  int pad_w_r = _param->common.pad_w_r;

  const int block_n = in_spec->shape[0];
  const int block_c = align_up(in_spec->shape[1], c_align);
  const int block_oh = out_spec->shape[2];
  const int block_iw = in_spec->shape[3];
  const int block_ow = out_spec->shape[3];
  const int block_ih = in_spec->shape[2];

  auto ret =
      avg_pool_2d_local(output_addr, input_addr, kh, kw, stride_h, stride_w,
                        pad_h_t, pad_h_b, pad_w_l, pad_w_r, block_n, block_c,
                        block_oh, block_iw, block_ow, block_ih);

  CHECK_PPL_RET(ret);
  if (ret != 0) {
    printf("\nblock [c:%d, oh:%d] vs total[n:%d, c:%d, "
           "oh:%d]\n",
           block_c, block_oh, in_spec->shape[0], in_spec->shape[1],
           out_spec->shape[2]);
    assert(0);
  }
}

#ifdef __cplusplus
}
#endif

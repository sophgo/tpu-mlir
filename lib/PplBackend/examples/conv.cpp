#include "fconv2d.h"
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

void api_conv_global(void *param, size_t param_size, void *input_spec,
                     void *output_spec, const char *chip, void *cmdid) {
  conv_global_spec_t *_param = (conv_global_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  auto weight_spec = in_spec + 1;
  auto bias_spec = in_spec + 2;

  // tiling
  int lane_num = 64;
  int eu_bytes = 64;
  int nic = 32;
  std::string chip_str(chip);
  if (chip_str == PPL_BM1688) {
    // TODO fix
    lane_num = 32;
    eu_bytes = 32;
  }
  int oc_align = lane_num;
  int ic_align = nic;
  int oh_align = 1;
  int ow_align = eu_bytes / 2;
  int n_align = 1;

  int n_secs = 1;
  int oc_secs = 1;
  int oh_secs = 1;
  int ow_secs = 1;
  int ic_secs = 1;

  int block_n = align_up(in_spec->shape[0] / n_secs, n_align);
  int block_oc = align_up(_param->common.output_c / oc_secs, oc_align);
  int block_oh = align_up(out_spec->shape[2] / oh_secs, oh_align);
  int block_ic = align_up(in_spec->shape[1] / ic_secs, ic_align);
  int block_ow = align_up(out_spec->shape[3] / ow_secs, ow_align);

  auto call_kernel = [&]() {
    return fconv2d(
        chip, cmdid, out_spec->addr, in_spec->addr, weight_spec->addr,
        bias_spec->addr, in_spec->shape[0], in_spec->shape[1],
        in_spec->shape[2], in_spec->shape[3], _param->common.output_c,
        _param->common.has_bias, _param->common.if_relu, _param->common.kh,
        _param->common.kw, _param->common.stride_h, _param->common.stride_w,
        _param->common.dh, _param->common.dw, _param->common.pad_h_t,
        _param->common.pad_h_b, _param->common.pad_w_l, _param->common.pad_w_r,
        block_n, block_oc, block_oh, block_ic, block_ow);
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
        // if (ret == -1) {
        //   printf("local mem not enough, reduce block size");
        //   assert(0);
        // }
      }
    }
    return 1;
  };

  std::vector<std::tuple<int *, int, int>> value;
  // split order n->oc->oh->ic->ow
  value.emplace_back(&block_n, in_spec->shape[0], n_align);
  value.emplace_back(&block_oc, _param->common.output_c, oc_align);
  value.emplace_back(&block_oh, out_spec->shape[2], oh_align);
  value.emplace_back(&block_ic, in_spec->shape[1], ic_align);
  value.emplace_back(&block_ow, out_spec->shape[3], ow_align);

  int ret = split_func(value);
  printf("block [n:%d, oc:%d, oh:%d, ic:%d, ow:%d] vs total[n:%d, oc:%d, "
         "oh:%d, ic:%d, ow:%d]\n",
         block_n, block_oc, block_oh, block_ic, block_ow, in_spec->shape[0],
         _param->common.output_c, out_spec->shape[2], in_spec->shape[1],
         out_spec->shape[3]);
  if (ret != 0) {
    assert(0 && "tiling failed\n");
  }
}

#ifdef __cplusplus
}
#endif

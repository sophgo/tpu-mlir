#include "avg_pool_2d_bf16.h"
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


void api_avgpool_global(void *param, size_t param_size, void *input_spec,
                     void *output_spec, const char *chip, void *cmdid) {
  pooling_common_spec_t *_param = (pooling_common_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;

  int lane_num = 64;
  int eu_bytes = 64;
  std::string chip_str(chip);
  if (chip_str == PPL_BM1688) {
    lane_num = 32;
    eu_bytes = 32;
  }
  int c_align = lane_num;
  int oh_align = 1;

  int c_secs = 1;
  int oh_secs = 1;

  int merged_c = in_spec->shape[1] * in_spec->shape[0];
  int block_c = align_up(merged_c / c_secs, c_align);
  int block_oh = align_up(out_spec->shape[2] / oh_secs, oh_align);

  auto call_kernel = [&]() {
    return avg_pool_2d_bf16(
        chip, cmdid, out_spec->addr, in_spec->addr, in_spec->shape[0],
        in_spec->shape[1], _param->kh, _param->kw, _param->stride_h,
        _param->stride_w, _param->pad_h_t, _param->pad_h_b, _param->pad_w_l,
        _param->pad_w_r, in_spec->shape[2], in_spec->shape[3],
        out_spec->shape[2], out_spec->shape[3], block_c, block_oh);
  };
  auto split_func = [&](std::vector<std::tuple<int *, int, int>> &value) {
    for (int i = 0; i < value.size(); i++) {
      auto item = value[i];
      int max_val = std::get<1>(item);
      int start_secs = i == 1 ? 2 : 1; // only enable pipeline in inner loop
      for (int secs = start_secs; secs < max_val; ++secs) {
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
  value.emplace_back(&block_c, merged_c, c_align);
  value.emplace_back(&block_oh, out_spec->shape[2], oh_align);

  int ret = split_func(value);
  printf("block [c:%d, oh:%d] vs total[n:%d, c:%d, "
         "oh:%d]\n",
         block_c, block_oh, in_spec->shape[0],
         in_spec->shape[1], out_spec->shape[2]);


  if (ret != 0) {
    assert(0 && "tiling failed\n");
  }
}

#ifdef __cplusplus
}
#endif

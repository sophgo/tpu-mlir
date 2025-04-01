#include "interp.h"
#include "helper.h"
#include "tpu_mlir/Backend/BM168x/Param.h"
#include <algorithm> // for std::clamp
#include <assert.h>
#include <cmath> // for std::floor
#include <cstdio>
#include <functional>
#include <iostream>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

using INTERP =
    std::function<int(unsigned long long v1,
                      unsigned long long v2, unsigned long long v3, int32_t v11,
                      int32_t v12, int32_t v13, float v14, int32_t v15,
                      int32_t v16, int32_t v17, int32_t v18, int32_t v19)>;

void api_interp_global(void *param, size_t param_size, void *input_spec,
                       void *output_spec) {

  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  interp_global_param_t *_param = (interp_global_param_t *)param;
  const int core_num = get_core_num();

  auto N = in_spec->shape[0];
  auto C = in_spec->shape[1];
  auto H_in = in_spec->shape[2];
  auto W_in = in_spec->shape[3];
  auto H_out = out_spec->shape[2];
  auto W_out = out_spec->shape[3];

  // tpu_local_mem_size_per_npu = 262144bit = 256KB
  int npu_size = 0;
  int npu_num = 0;
  int dtype_size = 4;
  auto block_h = 0;
  int ret = 0;
  INTERP func = nullptr;
  bool is_fp16 = in_spec[0].dtype == DTYPE_FP16;
  bool is_bf16 = in_spec[0].dtype == DTYPE_BFP16;
  func = is_fp16 ? interp_fp16 : is_bf16 ? interp_bf16 : interp_fp32;
  std::string chip_str = get_chip_str();
  if (chip_str == PPL_BM1688) {
    npu_size = 128 * 1024;
    npu_num = 32;
  } else if (chip_str == PPL_BM1684X) {
    npu_size = 256 * 1024;
    npu_num = 64;
  }
  block_h = static_cast<int>(std::floor(
      npu_size / dtype_size / (W_in + W_out * 16 / npu_num + W_out * 6)));
  if (H_out < block_h) {
    block_h = H_out;
  }

  while (block_h > 0) {
    printf("block_h:%d\n", block_h);
    ret = func(out_spec->addr, in_spec[0].addr,
               _param->spec.buffer_addr, core_num, N, C, H_in, W_in, H_out,
               W_out, block_h, _param->spec.common.align_corners);
    CHECK_PPL_RET(ret);
    if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr) {
      printf("block is not suitable, have another try !!!\n");
      block_h -= 1;
      continue;
    }
    break;
  }
  if (block_h <= 0) {
    printf("Error: block split failed!!!\n");
    exit(-1);
  }
  printf("interp success!!\n");
}

#ifdef __cplusplus
}
#endif

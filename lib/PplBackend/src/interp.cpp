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
    std::function<int(const char *chip, void *pid_node, unsigned long long v1,
                      unsigned long long v2, unsigned long long v3, int32_t v11,
                      int32_t v12, int32_t v13, float v14, int32_t v15,
                      int32_t v16, int32_t v17, int32_t v18)>;

void api_interp_global(void *param, size_t param_size, void *input_spec,
                       void *output_spec, const int core_num, const char *chip,
                       void *cmdid) {

  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  interp_global_param_t *_param = (interp_global_param_t *)param;

  auto N = in_spec->shape[0];
  auto C = in_spec->shape[1];
  auto H_in = in_spec->shape[2];
  auto W_in = in_spec->shape[3];
  auto H_out = out_spec->shape[2];
  auto W_out = out_spec->shape[3];

  // tpu_local_mem_size_per_npu = 262144bit = 256KB
  // tpu_bank_num() = 16;
  // int bank_size = tpu_local_mem_size_per_npu() / tpu_bank_num(); //16KB
  int bank_size = 0;
  int dtype_size = 0;
  auto block_h = 0;
  int ret = 0;
  INTERP func = nullptr;
  bool is_fp16 = in_spec[0].dtype == DTYPE_FP16;
  bool is_bf16 = in_spec[0].dtype == DTYPE_BFP16;
  func = is_fp16 ? interp_fp16 : is_bf16 ? interp_bf16 : interp_fp32;
  // dtype_size = is_fp16 ? 2 : 4;
  dtype_size = (is_fp16 || is_bf16) ? 2 : 4;
  std::string chip_str(chip);
  if (chip_str == PPL_BM1688) {
    bank_size = 8 * 1024;
  } else if (chip_str == PPL_BM1684X) {
    bank_size = 16 * 1024;
  }
  int output_per_c_size = H_out * W_out * dtype_size;
  if (output_per_c_size < bank_size) {
    block_h = H_out;
  } else {
    block_h = static_cast<int>(std::floor(bank_size / W_out / dtype_size));
  }

  while (block_h > 0) {
    ret = func(chip, cmdid, out_spec->addr, in_spec[0].addr,
               _param->spec.buffer_addr, core_num, N, C, H_in, W_in, H_out,
               W_out, block_h);
    CHECK_PPL_RET(ret);
    if (ret == PplLocalAddrAssignErr) {
      return;
    }
    break;
  }
}

#ifdef __cplusplus
}
#endif

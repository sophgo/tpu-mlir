#include "helper.h"
#include "interp_linear.h"
#include "interp_nearest.h"
#include "ppl_static_host.h"
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

using INTERP = std::function<int(
    unsigned long long v1, unsigned long long v2, int32_t v11, int32_t v12,
    int32_t v13, float v14, int32_t v15, int32_t v16, int32_t v17, int32_t v18,
    int32_t v19, int32_t v20)>;

using fill_buffer_func = std::function<int(
    unsigned long long ptr_output_v1, unsigned long long ptr_input_v2,
    int32_t g_core_num_v4, int32_t N_v5, int32_t C_v6, float H_in_v7,
    int32_t W_in_v8, int32_t H_out_v9, int32_t W_out_v10, int32_t block_h_v11,
    int32_t block_w_v12, int32_t align_corners_v13, void *buffer)>;

static INTERP get_interp_func(bool is_fp16, bool is_bf16, bool linear_mode) {
  if (linear_mode) {
    return is_fp16   ? interp_linear_fp16
           : is_bf16 ? interp_linear_bf16
                     : interp_linear_fp32;
  } else {
    return is_fp16   ? interp_nearest_fp16
           : is_bf16 ? interp_nearest_bf16
                     : interp_nearest_fp32;
  }
}

static fill_buffer_func get_fill_interp_buffer(bool is_fp16, bool is_bf16,
                                               bool linear_mode) {
  if (linear_mode) {
    return is_fp16   ? fill_interp_linear_fp16_struct
           : is_bf16 ? fill_interp_linear_bf16_struct
                     : fill_interp_linear_fp32_struct;
  } else {
    return is_fp16   ? fill_interp_nearest_fp16_struct
           : is_bf16 ? fill_interp_nearest_bf16_struct
                     : fill_interp_nearest_fp32_struct;
  }
}

typedef struct {
  int num_dims;
  int N;
  int C;
  int H_in;
  int W_in;
  int H_out;
  int W_out;
} TensorDims;

// The function for extracting spatial information
TensorDims extract_tensor_dims(const tensor_spec_t *in_spec,
                               const tensor_spec_t *out_spec) {
  TensorDims dims = {0};
  for (int i = 0; i < 8; i++) {
    if (in_spec->shape[i]) {
      dims.num_dims++;
    } else {
      break;
    }
  }
  // Initialize default dimension values
  dims.N = dims.C = dims.H_in = dims.W_in = dims.H_out = dims.W_out = 1;
  // Extract spatial information based on dimension positions
  if (dims.num_dims > 0) {
    dims.W_in = in_spec->shape[dims.num_dims - 1];
    dims.W_out = out_spec->shape[dims.num_dims - 1];
  }
  if (dims.num_dims > 1) {
    dims.H_in = in_spec->shape[dims.num_dims - 2];
    dims.H_out = out_spec->shape[dims.num_dims - 2];
  }
  if (dims.num_dims > 2) {
    dims.C = in_spec->shape[dims.num_dims - 3];
  }
  if (dims.num_dims > 3) {
    dims.N = in_spec->shape[dims.num_dims - 4];
  }
  // Handle high-dimensional data (dim > 4)
  for (int i = 4; i < dims.num_dims; i++) {
    dims.N *= in_spec->shape[dims.num_dims - i - 1];
  }

  return dims;
}

// static interface
void api_interp_global(void *param, size_t param_size, void *input_spec,
                       void *output_spec) {

  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  interp_global_param_t *_param = (interp_global_param_t *)param;
  const int core_num = get_core_num();

  // Call the extraction function
  TensorDims dims = extract_tensor_dims(in_spec, out_spec);
  int N = dims.N;
  int C = dims.C;
  int H_in = dims.H_in;
  int W_in = dims.W_in;
  int H_out = dims.H_out;
  int W_out = dims.W_out;

  // tpu_local_mem_size_per_npu = 262144bit = 256KB
  int npu_size = 0;
  int npu_num = 0;
  int dtype_size = 4;
  auto block_h = 0;
  auto block_h_origin = 0;
  auto block_w = W_out;
  int ret = 0;
  bool is_fp16 = in_spec[0].dtype == DTYPE_FP16;
  bool is_bf16 = in_spec[0].dtype == DTYPE_BFP16;
  bool linear_mode = true;
  int align_corners = 0;
  if (_param->spec.common.platform_sp == ONNX_NEAREST ||
      _param->spec.common.platform_sp == PYTORCH_NEAREST) {
    linear_mode = false;
  }
  if (_param->spec.common.align_corners) {
    align_corners = 1;
  }
  if (_param->spec.common.platform_sp == PYTORCH_NEAREST) {
    align_corners = 0;
  }

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
  INTERP func = get_interp_func(is_fp16, is_bf16, linear_mode);
  block_h_origin = block_h;
  while (block_h > 0) {
    ret = func(out_spec->addr, in_spec[0].addr, core_num, N, C, H_in, W_in,
               H_out, W_out, block_h, block_w, align_corners);
    CHECK_PPL_RET(ret);
    if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr) {
      printf("block_h is not suitable, have another try !!!\n");
      block_h -= 1;
      continue;
    }
    break;
  }
  if (block_h <= 0) {
    while (block_w > 0) {
      if (block_h == 0) {
        block_h = 1;
      } else {
        block_h = block_h_origin;
      }
      block_w = block_w / 2;
      ret = func(out_spec->addr, in_spec[0].addr, core_num, N, C, H_in, W_in,
                 H_out, W_out, block_h, block_w, align_corners);
      if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr) {
        printf("block_w is not suitable, have another try !!!\n");
        block_h -= 1;
        continue;
      }
      break;
    }
    if (block_w < 0) {
      printf("Error: block_w split failed!!!\n");
      exit(-1);
    }
  }
  printf("interp success!!\n");
}

// dynamic interface
int api_dyn_interp_global(void *param, void *input_spec, void *output_spec,
                          void *buffer) {
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  interp_global_param_t *_param = (interp_global_param_t *)param;
  const int core_num = get_core_num();

  // Call the extraction function
  TensorDims dims = extract_tensor_dims(in_spec, out_spec);
  int N = dims.N;
  int C = dims.C;
  int H_in = dims.H_in;
  int W_in = dims.W_in;
  int H_out = dims.H_out;
  int W_out = dims.W_out;

  // tpu_local_mem_size_per_npu = 262144bit = 256KB
  int npu_size = 0;
  int npu_num = 0;
  int dtype_size = 4;
  auto block_h = 0;
  auto block_h_origin = 0;
  auto block_w = W_out;
  int ret = 0;

  bool is_fp16 = in_spec[0].dtype == DTYPE_FP16;
  bool is_bf16 = in_spec[0].dtype == DTYPE_BFP16;
  bool linear_mode = true;
  int align_corners = 0;
  if (_param->spec.common.platform_sp == ONNX_NEAREST ||
      _param->spec.common.platform_sp == PYTORCH_NEAREST) {
    linear_mode = false;
  }
  if (_param->spec.common.align_corners) {
    align_corners = 1;
  }
  if (_param->spec.common.platform_sp == PYTORCH_NEAREST) {
    align_corners = 0;
  }
  if (buffer) {
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
    INTERP func = get_interp_func(is_fp16, is_bf16, linear_mode);
    block_h_origin = block_h;

    while (block_h > 0) {
      ret = func(out_spec->addr, in_spec[0].addr, core_num, N, C, H_in, W_in,
                 H_out, W_out, block_h, block_w, align_corners);
      CHECK_PPL_RET(ret);
      if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr) {
        printf("block is not suitable, have another try !!!\n");
        block_h -= 1;
        continue;
      }
      break;
    }
    if (block_h <= 0) {
      while (block_w > 0) {
        if (block_h == 0) {
          block_h = 1;
        } else {
          block_h = block_h_origin;
        }
        block_w = block_w / 2;
        ret = func(out_spec->addr, in_spec[0].addr, core_num, N, C, H_in, W_in,
                   H_out, W_out, block_h, block_w, align_corners);
        if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr) {
          printf("block_w is not suitable, have another try !!!\n");
          block_h -= 1;
          continue;
        }
        break;
      }
      if (block_w < 0) {
        printf("Error: block_w split failed!!!\n");
        exit(-1);
      }
    }
    printf("interp success!!\n");
  }
  fill_buffer_func fill_func =
      get_fill_interp_buffer(is_fp16, is_bf16, linear_mode);
  return fill_func(out_spec->addr, in_spec[0].addr, core_num, N, C, H_in, W_in,
                   H_out, W_out, block_h, block_w, align_corners, buffer);
}

#ifdef __cplusplus
}
#endif

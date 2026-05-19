//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "ppl_dyn_fw.h"
#include "w8a8_block_matmul.c"

// global
void dynamic_glb_w8a8_block_matmul_ctrl(
    void *ctx, void *param, global_tensor_spec_t *input_spec,
    global_tensor_spec_t *output_spec) {

  // Input spec layout: input_spec[0]=activation,
  //                    input_spec[1]=weight (fp8e4m3),
  //                    input_spec[2]=weight_scale.
  // Activation/output use [G, M, 1, K] / [G, M, 1, N], weight is [G, N, 1, K].
  int32_t data_type = input_spec[0].dtype;
  output_spec[0].dtype = data_type;
  output_spec[0].dims = input_spec[0].dims;
  memcpy(output_spec[0].shape, input_spec[0].shape,
         input_spec[0].dims * sizeof(int));
  output_spec[0].shape[3] = input_spec[1].shape[1]; // N
  output_spec[0].elem_num = output_spec[0].shape[0] * output_spec[0].shape[1] *
                            output_spec[0].shape[2] * output_spec[0].shape[3];

  if (data_type == FW_DTYPE_FP16) {
    tpu_kernel_api_w8a8_block_matmul_f16_t *param_ =
        (tpu_kernel_api_w8a8_block_matmul_f16_t *)param;
    param_->ptr_in = input_spec[0].addr;
    param_->ptr_weight = input_spec[1].addr;
    param_->ptr_weight_scale = input_spec[2].addr;
    param_->ptr_out = output_spec[0].addr;
    param_->G = input_spec[0].shape[0];
    param_->M = input_spec[0].shape[1];
    param_->K = input_spec[0].shape[3];
    param_->N = input_spec[1].shape[1];
    if (param_->core_num > tpu_core_num()) {
      param_->core_num = tpu_core_num();
    }
    w8a8_block_matmul_f16_entry(param_);
  } else if (data_type == FW_DTYPE_BFP16) {
    tpu_kernel_api_w8a8_block_matmul_bf16_t *param_ =
        (tpu_kernel_api_w8a8_block_matmul_bf16_t *)param;
    param_->ptr_in = input_spec[0].addr;
    param_->ptr_weight = input_spec[1].addr;
    param_->ptr_weight_scale = input_spec[2].addr;
    param_->ptr_out = output_spec[0].addr;
    param_->G = input_spec[0].shape[0];
    param_->M = input_spec[0].shape[1];
    param_->K = input_spec[0].shape[3];
    param_->N = input_spec[1].shape[1];
    if (param_->core_num > tpu_core_num()) {
      param_->core_num = tpu_core_num();
    }
    w8a8_block_matmul_bf16_entry(param_);
  }
}

REGISTER_PPL_DYN_OP(PPL_FW_W8A8_BLOCK_MATMUL,
                    dynamic_glb_w8a8_block_matmul_ctrl, 0);

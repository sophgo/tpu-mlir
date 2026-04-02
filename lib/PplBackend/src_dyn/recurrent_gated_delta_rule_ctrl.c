//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "ppl_dyn_fw.h"
#include "recurrent_gated_delta_rule.c"

// global
void dynamic_glb_recurrent_gated_delta_rule_ctrl(
    void *ctx, void *param, global_tensor_spec_t *input_spec,
    global_tensor_spec_t *output_spec) {

  int32_t data_type = input_spec[0].dtype;
  output_spec[0].dtype = data_type;
  memcpy(output_spec[0].shape, input_spec[2].shape,
         input_spec[2].dims * sizeof(int));
  output_spec[0].dims = 4;
  output_spec[0].shape[0] = input_spec[0].shape[0];
  output_spec[0].shape[1] = 1;
  output_spec[0].elem_num = input_spec[2].elem_num;
  if (data_type == FW_DTYPE_FP16) {
    tpu_kernel_api_recurrent_gated_delta_rule_f16_t *param_ =
        (tpu_kernel_api_recurrent_gated_delta_rule_f16_t *)param;
    param_->ptr_Q = input_spec[0].addr;
    param_->ptr_K = input_spec[1].addr;
    param_->ptr_V = input_spec[2].addr;
    param_->ptr_g = input_spec[3].addr;
    param_->ptr_beta = input_spec[4].addr;
    param_->ptr_last_recurrent_state = input_spec[5].addr;
    param_->ptr_core_attn_out = output_spec[0].addr;
    param_->B = input_spec[0].shape[0];
    if (param_->core_num > tpu_core_num()) {
      param_->core_num = tpu_core_num();
    }
    output_spec[0].shape[2] = param_->num_v_heads;
    output_spec[0].shape[3] = param_->d;
    recurrent_gated_delta_rule_f16_entry(param_);
  } else if (data_type == FW_DTYPE_BFP16) {
    tpu_kernel_api_recurrent_gated_delta_rule_bf16_t *param_ =
        (tpu_kernel_api_recurrent_gated_delta_rule_bf16_t *)param;
    param_->ptr_Q = input_spec[0].addr;
    param_->ptr_K = input_spec[1].addr;
    param_->ptr_V = input_spec[2].addr;
    param_->ptr_g = input_spec[3].addr;
    param_->ptr_beta = input_spec[4].addr;
    param_->ptr_last_recurrent_state = input_spec[5].addr;
    param_->ptr_core_attn_out = output_spec[0].addr;
    param_->B = input_spec[0].shape[0];
    if (param_->core_num > tpu_core_num()) {
      param_->core_num = tpu_core_num();
    }
    output_spec[0].shape[2] = param_->num_v_heads;
    output_spec[0].shape[3] = param_->d;
    recurrent_gated_delta_rule_bf16_entry(param_);
  }
}

REGISTER_PPL_DYN_OP(PPL_FW_RECURRENT_GATED_DELTA_RULE,
                    dynamic_glb_recurrent_gated_delta_rule_ctrl, 0);

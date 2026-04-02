//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "chunk_gated_delta_rule.c"
#include "ppl_dyn_fw.h"

// global
void dynamic_glb_chunk_gated_delta_rule_ctrl(
    void *ctx, void *param, global_tensor_spec_t *input_spec,
    global_tensor_spec_t *output_spec) {

  int32_t data_type = input_spec[2].dtype;
  output_spec[0].dtype = data_type;
  memcpy(output_spec[0].shape, input_spec[2].shape,
         input_spec[2].dims * sizeof(int));
  output_spec[0].dims = input_spec[2].dims;
  output_spec[0].elem_num = input_spec[2].elem_num;
  if (data_type == FW_DTYPE_FP16) {
    tpu_kernel_api_chunk_gated_delta_rule_f16_t *param_ =
        (tpu_kernel_api_chunk_gated_delta_rule_f16_t *)param;
    param_->ptr_Q = input_spec[0].addr;
    param_->ptr_K = input_spec[1].addr;
    param_->ptr_V = input_spec[2].addr;
    param_->ptr_g = input_spec[3].addr;
    param_->ptr_beta = input_spec[4].addr;
    param_->ptr_last_recurrent_state = input_spec[5].addr;
    param_->ptr_triu_mask = input_spec[6].addr;
    param_->ptr_strict_triu_mask = input_spec[7].addr;
    param_->ptr_tril_mask = input_spec[8].addr;
    param_->ptr_eye = input_spec[9].addr;
    param_->ptr_core_attn_out = output_spec[0].addr;
    param_->B = input_spec[0].shape[0];
    param_->S = input_spec[0].shape[1];
    if (param_->core_num > tpu_core_num()) {
      param_->core_num = tpu_core_num();
    }
    chunk_gated_delta_rule_f16_entry(param_);
  } else if (data_type == FW_DTYPE_BFP16) {
    // similar assignment for bfloat16
    tpu_kernel_api_chunk_gated_delta_rule_bf16_t *param_ =
        (tpu_kernel_api_chunk_gated_delta_rule_bf16_t *)param;
    param_->ptr_Q = input_spec[0].addr;
    param_->ptr_K = input_spec[1].addr;
    param_->ptr_V = input_spec[2].addr;
    param_->ptr_g = input_spec[3].addr;
    param_->ptr_beta = input_spec[4].addr;
    param_->ptr_last_recurrent_state = input_spec[5].addr;
    param_->ptr_triu_mask = input_spec[6].addr;
    param_->ptr_strict_triu_mask = input_spec[7].addr;
    param_->ptr_tril_mask = input_spec[8].addr;
    param_->ptr_eye = input_spec[9].addr;
    param_->ptr_core_attn_out = output_spec[0].addr;
    param_->B = input_spec[0].shape[0];
    param_->S = input_spec[0].shape[1];
    if (param_->core_num > tpu_core_num()) {
      param_->core_num = tpu_core_num();
    }
    chunk_gated_delta_rule_bf16_entry(param_);
  }
}

REGISTER_PPL_DYN_OP(PPL_FW_CHUNK_GATED_DELTA_RULE,
                    dynamic_glb_chunk_gated_delta_rule_ctrl, 0);

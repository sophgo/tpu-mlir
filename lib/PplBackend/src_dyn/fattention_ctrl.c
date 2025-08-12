#include "ppl_dyn_fw.h"
#include "fattention_bf16.c"
#include "fattention_fp16.c"
#include "fattention_v2.c"


// global
void dynamic_glb_flash_attention_layer_ctrl(void *ctx, void *param,
                                            global_tensor_spec_t *input_spec,
                                            global_tensor_spec_t *output_spec) {
  output_spec->dtype = input_spec->dtype;
  output_spec->dims = 3;
  output_spec->shape[0] = input_spec->shape[0];
  output_spec->shape[1] = input_spec->shape[1];
  output_spec->shape[2] = input_spec->shape[2] * input_spec->shape[3];
  bool is_mha = input_spec->shape[2] == (input_spec + 1)->shape[2];
  if (input_spec->dtype == FW_DTYPE_FP16) {
    if (is_mha) {
      tpu_kernel_api_flash_attention_mha_f16_t *_param =
          (tpu_kernel_api_flash_attention_mha_f16_t *)param;
      _param->ptr_out = output_spec->addr;
      _param->ptr_q = input_spec[0].addr;
      _param->ptr_k = input_spec[1].addr;
      _param->ptr_v = input_spec[2].addr;
      _param->ptr_mask = input_spec[3].addr;
      _param->b = input_spec->shape[0];
      _param->qm = input_spec->shape[1];
      _param->kvm = (input_spec + 1)->shape[1];
      _param->d = input_spec->shape[3];
      _param->q_head = input_spec->shape[2];
      _param->kv_head = (input_spec + 1)->shape[2];
      flash_attention_mha_f16(_param);
    } else {
      tpu_kernel_api_flash_attention_gqa_f16_t *_param =
          (tpu_kernel_api_flash_attention_gqa_f16_t *)param;
      _param->ptr_out = output_spec->addr;
      _param->ptr_q = input_spec[0].addr;
      _param->ptr_k = input_spec[1].addr;
      _param->ptr_v = input_spec[2].addr;
      _param->ptr_mask = input_spec[3].addr;
      _param->b = input_spec->shape[0];
      _param->qm = input_spec->shape[1];
      _param->kvm = (input_spec + 1)->shape[1];
      _param->d = input_spec->shape[3];
      _param->q_head = input_spec->shape[2];
      _param->kv_head = (input_spec + 1)->shape[2];
      flash_attention_gqa_f16(_param);
    }
  } else if (input_spec->dtype == FW_DTYPE_BFP16) {
    if (is_mha) {
      tpu_kernel_api_flash_attention_mha_bf16_t *_param =
          (tpu_kernel_api_flash_attention_mha_bf16_t *)param;
      _param->ptr_out = output_spec->addr;
      _param->ptr_q = input_spec[0].addr;
      _param->ptr_k = input_spec[1].addr;
      _param->ptr_v = input_spec[2].addr;
      _param->ptr_mask = input_spec[3].addr;
      _param->b = input_spec->shape[0];
      _param->qm = input_spec->shape[1];
      _param->kvm = (input_spec + 1)->shape[1];
      _param->d = input_spec->shape[3];
      _param->q_head = input_spec->shape[2];
      _param->kv_head = (input_spec + 1)->shape[2];
      flash_attention_mha_bf16(_param);
    } else {
      tpu_kernel_api_flash_attention_gqa_bf16_t *_param =
          (tpu_kernel_api_flash_attention_gqa_bf16_t *)param;
      _param->ptr_out = output_spec->addr;
      _param->ptr_q = input_spec[0].addr;
      _param->ptr_k = input_spec[1].addr;
      _param->ptr_v = input_spec[2].addr;
      _param->ptr_mask = input_spec[3].addr;
      _param->b = input_spec->shape[0];
      _param->qm = input_spec->shape[1];
      _param->kvm = (input_spec + 1)->shape[1];
      _param->d = input_spec->shape[3];
      _param->q_head = input_spec->shape[2];
      _param->kv_head = (input_spec + 1)->shape[2];
      flash_attention_gqa_bf16(_param);
    }
  }
}

void dynamic_glb_flash_attention_heigh_prec_layer_ctrl(void *ctx,
    void *param, global_tensor_spec_t *input_spec,
    global_tensor_spec_t *output_spec) {
  output_spec->dtype = input_spec->dtype;
  output_spec->dims = 3;
  output_spec->shape[0] = input_spec->shape[0];
  output_spec->shape[1] = input_spec->shape[1];
  output_spec->shape[2] = input_spec->shape[2] * input_spec->shape[3];
  bool is_mha = input_spec->shape[2] == (input_spec + 1)->shape[2];
  if (input_spec->dtype == FW_DTYPE_FP16) {
    if (is_mha) {
      tpu_kernel_api_flash_attention_mha_f16_high_precision_t *_param =
          (tpu_kernel_api_flash_attention_mha_f16_high_precision_t *)param;
      _param->ptr_out = output_spec->addr;
      _param->ptr_q = input_spec[0].addr;
      _param->ptr_k = input_spec[1].addr;
      _param->ptr_v = input_spec[2].addr;
      _param->ptr_mask = input_spec[3].addr;
      _param->b = input_spec->shape[0];
      _param->qm = input_spec->shape[1];
      _param->kvm = (input_spec + 1)->shape[1];
      _param->d = input_spec->shape[3];
      _param->q_head = input_spec->shape[2];
      _param->kv_head = (input_spec + 1)->shape[2];
      flash_attention_mha_f16_high_precision(_param);
    } else {
      tpu_kernel_api_flash_attention_gqa_f16_high_precision_t *_param =
          (tpu_kernel_api_flash_attention_gqa_f16_high_precision_t *)param;
      _param->ptr_out = output_spec->addr;
      _param->ptr_q = input_spec[0].addr;
      _param->ptr_k = input_spec[1].addr;
      _param->ptr_v = input_spec[2].addr;
      _param->ptr_mask = input_spec[3].addr;
      _param->b = input_spec->shape[0];
      _param->qm = input_spec->shape[1];
      _param->kvm = (input_spec + 1)->shape[1];
      _param->d = input_spec->shape[3];
      _param->q_head = input_spec->shape[2];
      _param->kv_head = (input_spec + 1)->shape[2];
      flash_attention_gqa_f16_high_precision(_param);
    }
  } else if (input_spec->dtype == FW_DTYPE_BFP16) {
    if (is_mha) {
      tpu_kernel_api_flash_attention_mha_bf16_high_precision_t *_param =
          (tpu_kernel_api_flash_attention_mha_bf16_high_precision_t *)param;
      _param->ptr_out = output_spec->addr;
      _param->ptr_q = input_spec[0].addr;
      _param->ptr_k = input_spec[1].addr;
      _param->ptr_v = input_spec[2].addr;
      _param->ptr_mask = input_spec[3].addr;
      _param->b = input_spec->shape[0];
      _param->qm = input_spec->shape[1];
      _param->kvm = (input_spec + 1)->shape[1];
      _param->d = input_spec->shape[3];
      _param->q_head = input_spec->shape[2];
      _param->kv_head = (input_spec + 1)->shape[2];
      flash_attention_mha_bf16_high_precision(_param);
    } else {
      tpu_kernel_api_flash_attention_gqa_bf16_high_precision_t *_param =
          (tpu_kernel_api_flash_attention_gqa_bf16_high_precision_t *)param;
      _param->ptr_out = output_spec->addr;
      _param->ptr_q = input_spec[0].addr;
      _param->ptr_k = input_spec[1].addr;
      _param->ptr_v = input_spec[2].addr;
      _param->ptr_mask = input_spec[3].addr;
      _param->b = input_spec->shape[0];
      _param->qm = input_spec->shape[1];
      _param->kvm = (input_spec + 1)->shape[1];
      _param->d = input_spec->shape[3];
      _param->q_head = input_spec->shape[2];
      _param->kv_head = (input_spec + 1)->shape[2];
      flash_attention_gqa_bf16_high_precision(_param);
    }
  }
}
REGISTER_PPL_DYN_OP(PPL_FW_FLASH_ATTENTION,
                    dynamic_glb_flash_attention_layer_ctrl, 0);
REGISTER_PPL_DYN_OP(PPL_FW_FLASH_ATTENTION_HEIGH_PRECISION,
                    dynamic_glb_flash_attention_heigh_prec_layer_ctrl, 0);
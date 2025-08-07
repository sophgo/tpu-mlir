#include "ppl_dyn_fw.h"
#include "add_const_fp.c"
#include "add_const_fp_local.c"

// global
void dynamic_glb_add_const_fp_layer_ctrl(void *ctx, void *param,
                                         global_tensor_spec_t *input_spec,
                                         global_tensor_spec_t *output_spec) {
  output_spec->dtype = input_spec->dtype;
  memcpy(output_spec->shape, input_spec->shape, input_spec->dims * sizeof(int));
  output_spec->dims = input_spec->dims;
  output_spec->elem_num = input_spec->elem_num;

  if (input_spec->dtype == FW_DTYPE_FP16) {
    tpu_kernel_api_add_const_f16_t *_param =
        (tpu_kernel_api_add_const_f16_t *)param;
    _param->ptr_dst = output_spec->addr;
    _param->ptr_src = input_spec->addr;
    _param->N = input_spec->shape[0];
    _param->C = input_spec->shape[1];
    _param->H = input_spec->shape[2];
    _param->W = input_spec->shape[3];
    add_const_f16(_param);
  } else if (input_spec->dtype == FW_DTYPE_BFP16) {
    tpu_kernel_api_add_const_bf16_t *_param =
        (tpu_kernel_api_add_const_bf16_t *)param;
    _param->ptr_dst = output_spec->addr;
    _param->ptr_src = input_spec->addr;
    _param->N = input_spec->shape[0];
    _param->C = input_spec->shape[1];
    _param->H = input_spec->shape[2];
    _param->W = input_spec->shape[3];
    add_const_bf16(_param);
  } else if (input_spec->dtype == FW_DTYPE_FP32) {
    tpu_kernel_api_add_const_f32_t *_param =
        (tpu_kernel_api_add_const_f32_t *)param;
    _param->ptr_dst = output_spec->addr;
    _param->ptr_src = input_spec->addr;
    _param->N = input_spec->shape[0];
    _param->C = input_spec->shape[1];
    _param->H = input_spec->shape[2];
    _param->W = input_spec->shape[3];
    add_const_f32(_param);
  }
}

// local
void dynamic_add_const_fp_layer_ctrl(void *ctx, void *param,
                                     local_sec_info_t *sec_info,
                                     local_output_info_t *top,
                                     dynamic_local_tensor_info_t *tensor_info) {
  int shape[4];
  local_tensor_spec_t *output_spec = tensor_info->out_specs;
  local_tensor_spec_t *input_spec = tensor_info->in_specs;
  parse_input_slice_shape(sec_info, input_spec, shape);
  if (input_spec->dtype == FW_DTYPE_FP16) {
    tpu_kernel_api_add_const_f16_local_t *_param =
        (tpu_kernel_api_add_const_f16_local_t *)param;
    _param->o_local_ddr = output_spec->addr;
    _param->i_local_ddr = input_spec->addr;
    _param->N = shape[0];
    _param->C = shape[1];
    _param->H = shape[2];
    _param->W = shape[3];
    add_const_f16_local(_param);
  } else if (input_spec->dtype == FW_DTYPE_BFP16) {
    tpu_kernel_api_add_const_bf16_local_t *_param =
        (tpu_kernel_api_add_const_bf16_local_t *)param;
    _param->o_local_ddr = output_spec->addr;
    _param->i_local_ddr = input_spec->addr;
    _param->N = shape[0];
    _param->C = shape[1];
    _param->H = shape[2];
    _param->W = shape[3];
    add_const_bf16_local(_param);
  } else if (input_spec->dtype == FW_DTYPE_FP32) {
    tpu_kernel_api_add_const_f32_local_t *_param =
        (tpu_kernel_api_add_const_f32_local_t *)param;
    _param->o_local_ddr = output_spec->addr;
    _param->i_local_ddr = input_spec->addr;
    _param->N = shape[0];
    _param->C = shape[1];
    _param->H = shape[2];
    _param->W = shape[3];
    add_const_f32_local(_param);
  }
  set_out_info(top, tensor_info->in_refs, tensor_info->out_refs, sec_info);
}
REGISTER_PPL_DYN_OP(PPL_FW_ADD_CONST, dynamic_glb_add_const_fp_layer_ctrl,
                    dynamic_add_const_fp_layer_ctrl);
#include "ppl_dyn_fw.h"
#include "interp_linear.c"
#include "interp_nearest.c"

// global
void dynamic_glb_interp_linear_layer_ctrl(void *ctx, void *param,
                                          global_tensor_spec_t *input_spec,
                                          global_tensor_spec_t *output_spec) {
  output_spec->dtype = input_spec->dtype;
  output_spec->dims = input_spec->dims;
  int out_h, out_w;
  int in_h = input_spec->shape[2];
  int in_w = input_spec->shape[3];
  if (input_spec->dtype == FW_DTYPE_FP32) {
    tpu_kernel_api_interp_linear_fp32_t *_param =
        (tpu_kernel_api_interp_linear_fp32_t *)param;
    double h_ratio = (double)_param->H_out / _param->H_in;
    double w_ratio = (double)_param->W_out / _param->W_in;
    out_h = h_ratio * in_h + 1e-3;
    out_w = w_ratio * in_w + 1e-3;
    _param->ptr_output = output_spec->addr;
    _param->ptr_input = input_spec->addr;
    _param->H_in = in_h;
    _param->W_in = in_w;
    _param->H_out = out_h;
    _param->W_out = out_w;
    interp_linear_fp32(_param);
  } else if (input_spec->dtype == FW_DTYPE_BFP16) {
    tpu_kernel_api_interp_linear_bf16_t *_param =
        (tpu_kernel_api_interp_linear_bf16_t *)param;
    double h_ratio = (double)_param->H_out / _param->H_in;
    double w_ratio = (double)_param->W_out / _param->W_in;
    out_h = h_ratio * in_h + 1e-3;
    out_w = w_ratio * in_w + 1e-3;
    _param->ptr_output = output_spec->addr;
    _param->ptr_input = input_spec->addr;
    _param->H_in = in_h;
    _param->W_in = in_w;
    _param->H_out = out_h;
    _param->W_out = out_w;
    interp_linear_bf16(_param);
  } else if (input_spec->dtype == FW_DTYPE_FP16) {
    tpu_kernel_api_interp_linear_fp16_t *_param =
        (tpu_kernel_api_interp_linear_fp16_t *)param;
    float h_ratio = (float)_param->H_out / _param->H_in;
    float w_ratio = (float)_param->W_out / _param->W_in;
    out_h = h_ratio * in_h + 1e-3;
    out_w = w_ratio * in_w + 1e-3;
    _param->ptr_output = output_spec->addr;
    _param->ptr_input = input_spec->addr;
    _param->H_in = in_h;
    _param->W_in = in_w;
    _param->H_out = out_h;
    _param->W_out = out_w;
    interp_linear_fp16(_param);
  } else {
    TPUKERNEL_ERR("%s, interp_linear not support dtype=%d\n", __func__,
                  input_spec->dtype);
  }
  output_spec->shape[0] = input_spec->shape[0];
  output_spec->shape[1] = input_spec->shape[1];
  output_spec->shape[2] = out_h;
  output_spec->shape[3] = out_w;
}

void dynamic_glb_interp_nearest_layer_ctrl(void *ctx, void *param,
                                           global_tensor_spec_t *input_spec,
                                           global_tensor_spec_t *output_spec) {
  output_spec->dtype = input_spec->dtype;
  output_spec->dims = input_spec->dims;
  int out_h, out_w;
  int in_h = input_spec->shape[2];
  int in_w = input_spec->shape[3];
  if (input_spec->dtype == FW_DTYPE_FP32) {
    tpu_kernel_api_interp_nearest_fp32_t *_param =
        (tpu_kernel_api_interp_nearest_fp32_t *)param;
    float h_ratio = (float)_param->H_out / _param->H_in;
    float w_ratio = (float)_param->W_out / _param->W_in;
    out_h = h_ratio * in_h + 1e-3;
    out_w = w_ratio * in_w + 1e-3;
    _param->ptr_output = output_spec->addr;
    _param->ptr_input = input_spec->addr;
    _param->H_in = in_h;
    _param->W_in = in_w;
    _param->H_out = out_h;
    _param->W_out = out_w;
    interp_nearest_fp32(_param);
  } else if (input_spec->dtype == FW_DTYPE_BFP16) {
    tpu_kernel_api_interp_nearest_bf16_t *_param =
        (tpu_kernel_api_interp_nearest_bf16_t *)param;
    float h_ratio = (float)_param->H_out / _param->H_in;
    float w_ratio = (float)_param->W_out / _param->W_in;
    out_h = h_ratio * in_h + 1e-3;
    out_w = w_ratio * in_w + 1e-3;
    _param->ptr_output = output_spec->addr;
    _param->ptr_input = input_spec->addr;
    _param->H_in = in_h;
    _param->W_in = in_w;
    _param->H_out = out_h;
    _param->W_out = out_w;
    interp_nearest_bf16(_param);
  } else if (input_spec->dtype == FW_DTYPE_FP16) {
    tpu_kernel_api_interp_nearest_fp16_t *_param =
        (tpu_kernel_api_interp_nearest_fp16_t *)param;
    float h_ratio = (float)_param->H_out / _param->H_in;
    float w_ratio = (float)_param->W_out / _param->W_in;
    out_h = h_ratio * in_h + 1e-3;
    out_w = w_ratio * in_w + 1e-3;
    _param->ptr_output = output_spec->addr;
    _param->ptr_input = input_spec->addr;
    _param->H_in = in_h;
    _param->W_in = in_w;
    _param->H_out = out_h;
    _param->W_out = out_w;
    interp_nearest_fp16(_param);
  } else {
    TPUKERNEL_ERR("%s, interp_linear not support dtype=%d\n", __func__,
                  input_spec->dtype);
  }
  output_spec->shape[0] = input_spec->shape[0];
  output_spec->shape[1] = input_spec->shape[1];
  output_spec->shape[2] = out_h;
  output_spec->shape[3] = out_w;
}
REGISTER_PPL_DYN_OP(PPL_FW_INTERP_LINEAR, dynamic_glb_interp_linear_layer_ctrl,
                    0);
REGISTER_PPL_DYN_OP(PPL_FW_INTERP_NEAREST,
                    dynamic_glb_interp_nearest_layer_ctrl, 0);

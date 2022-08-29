//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Backend/BM168x/BM168x.h"

// clang-format off
typedef void (*tensor_align_move_gen_cmd)(int local_mem_start_addr, int local_mem_idx, uint64_t sys_mem_start_addr, int src_N, int src_C, int src_H, int src_W, int src_format, int direction, int transpose, CMD_ID_NODE *pid_node);
typedef void (*general_matrix_move_gen_cmd)(int local_mem_start_addr, int local_mem_idx, uint64_t sys_mem_start_addr, int sec_size, int row_num, int col_num, uint32_t row_stride, int src_format, int direction, int transpose, int result_add, CMD_ID_NODE *pid_node);
typedef void (*nodechip_conv_forward_local)(int bottom_local_offset, int weight_local_offset, int bias_local_offset, int top_local_offset, int imm_local_offset, int *bottom_dim, int *top_dim, int groups, int kh, int kw, int dh, int dw, int up_pad_h, int down_pad_h, int left_pad_w, int right_pad_w, int stride_h, int stride_w, int using_bias, int result_add, int if_relu, float relu_upper_limit, int unused_ht_for_input_tensor, int unused_hb_for_input_tensor, int unused_wl_for_input_tensor, int unused_wr_for_input_tensor, void *id_node);
typedef void (*nodechip_winograd_forward_local)(int bottom_local_offset, int weight_local_offset, int bias_local_offset, int top_local_offset, int imm_local_offset, int *bottom_dim, int *top_dim, int groups, int up_pad_h, int down_pad_h, int left_pad_w, int right_pad_w, int using_bias, int result_add, int if_relu, float relu_upper_limit, int use_winograd, int unused_ht_for_input_tensor, int unused_hb_for_input_tensor, int unused_wl_for_input_tensor, int unused_wr_for_input_tensor, void *id_node);
typedef void (*nodechip_pooling_fix8b_forward_local)(int bottom_local_offset, int weight_local_offset, int bias_local_offset, int top_local_offset, int *bottom_dim, int *top_dim, int kh, int kw, int pad_h_top, int pad_h_bottom, int pad_w_left, int pad_w_right, int stride_h, int stride_w, int ins0_w, int ins0_h, int is_avg_pooling, int avg_pooling_mode, int r_shift, int using_bias, int rshift_typ, int opd0_sign, int opd1_sign, int opd2_sign, int res0_sign, int if_relu, void *pid_node);
typedef void (*nodechip_conv_forward_local_fix8b)(int bottom_local_offset, int weight_local_offset, int bias_local_offset, int top_local_offset, int imm_local_offset, int *bottom_dim, int *top_dim, int groups, int kh, int kw, int dh, int dw, int up_pad_h, int down_pad_h, int left_pad_w, int right_pad_w, int stride_h, int stride_w, int using_bias, int result_add, int if_relu, int relu_upper_limit, int unused_ht_for_input_tensor, int unused_hb_for_input_tensor, int unused_wl_for_input_tensor, int unused_wr_for_input_tensor, int ins_h, int ins_w, int rshiftbits, int opd0_sign, int opd1_sign, int opd2_sign, bool if_ic_inner, int if_concat_scale, int concat_scale_val, int concat_scale_rshift, int concat_output_sign, void *id_node);
typedef void (*nodechip_conv_forward_local_fix16b)(int bottom_local_offset, int weight_local_offset, int bias_local_offset, int top_local_offset, int imm_local_offset, int *bottom_dim, int *top_dim, int groups, int kh, int kw, int dh, int dw, int up_pad_h, int down_pad_h, int left_pad_w, int right_pad_w, int stride_h, int stride_w, int using_bias, int result_add, int if_relu, int relu_upper_limit, int unused_ht_for_input_tensor, int unused_hb_for_input_tensor, int ins_h, int ins_w, int rshiftbits, int opd0_sign, int opd1_sign, int opd2_sign, bool if_ic_inner, void *id_node);
typedef void (*nodechip_winograd_forward_local_fix8b)(int bottom_local_offset, int weight_local_offset, int bias_local_offset, int top_local_offset, int imm_local_offset, int *bottom_dim, int *top_dim, int groups, int up_pad_h, int down_pad_h, int left_pad_w, int right_pad_w, int using_bias, int result_add, int if_relu, int relu_upper_limit, int use_winograd, int unused_ht_for_input_tensor, int unused_hb_for_input_tensor, int unused_wl_for_input_tensor, int unused_wr_for_input_tensor, int rshift_bits, int opd0_sign, int opd1_sign, int opd2_sign, int opd0_short_str, int if_concat_scale, int concat_scale_val, int concat_scale_rshift, int concat_output_sign, void *id_node);
typedef void (*nodechip_winograd_double_buffer_forward_local_fix8b)(int bottom_local_offset, uint64_t weight_global_offset, int weight_local_offset, int bias_local_offset, int top_local_offset, int imm_local_offset, int double_buffer_local_offset, int *bottom_dim, int *top_dim, int groups, int up_pad_h, int down_pad_h, int left_pad_w, int right_pad_w, int using_bias, int result_add, int if_relu, int relu_upper_limit, int use_winograd, int unused_ht_for_input_tensor, int unused_hb_for_input_tensor, int unused_wl_for_input_tensor, int unused_wr_for_input_tensor, int rshift_bits, int opd0_sign, int opd1_sign, int opd2_sign, int opd0_short_str, int if_concat_scale, int concat_scale_val, int concat_scale_rshift, int concat_output_sign, void *id_node, void *id_node_gdma);
typedef void (*nodechip_deconv_forward_local)(int bottom_local_offset, int weight_local_offset, int bias_local_offset, int top_local_offset, int *bottom_dim, int *top_dim, int groups, int kh, int kw, int dh, int dw, int up_pad_h, int down_pad_h, int left_pad_w, int right_pad_w, int ins_h, int ins_w, int using_bias, int result_add, int if_relu, void *id_node);
typedef void (*nodechip_deconv_fix16b_forward_local)(int bottom_local_offset, int weight_local_offset, int bias_local_offset, int top_local_offset, int imm_local_offset, int *bottom_dim, int *top_dim, int groups, int kh, int kw, int dh, int dw, int ins_h, int ins_w, int up_pad_h, int down_pad_h, int left_pad_w, int right_pad_w, int using_bias, int if_relu, float relu_upper_limit, int rshift_num, int bottom_sign, int weight_sign, int bias_sign, void *id_node);
typedef void (*nodechip_deconv_fix8b_forward_local)(int bottom_local_offset, int weight_local_offset, int bias_local_offset, int top_local_offset, int *bottom_dim, int *top_dim, int groups, int kh, int kw, int dh, int dw, int up_pad_h, int down_pad_h, int left_pad_w, int right_pad_w, int ins_h, int ins_w, int using_bias, int if_relu, int rshift_num, int bottom_sign, int weight_sign, int bias_sign, void *id_node);
typedef void (*nodechip_pooling_forward_local)(uint32_t bottom_local_offset, int top_local_offset, int *bottom_dim, int *top_dim, int kh, int kw, int up_pad_h, int down_pad_h, int left_pad_w, int right_pad_w, int stride_h, int stride_w, int is_avg_pooling, int avg_pooling_mode, void *id_node, int if_relu);
typedef void (*nodechip_upsample_forward_local)(uint32_t bottom_local_offset, int top_local_offset, int *bottom_dim, int *top_dim, int size, void *id_node, int if_relu);
typedef void (*nodechip_upsample_fix8b_forward_local)(uint32_t bottom_local_offset, int top_local_offset, int *bottom_dim, int *top_dim, int size, int rshift_typ, int opd0_sign, int opd1_sign, int opd2_sign, int res0_sign, void *id_node, int if_relu);
typedef void (*nodechip_lrn_forward_local)(int bottom_local_offset, int top_local_offset, int imm_buffer_local_offset, int *bottom_dim, float alpha, int size, float beta, float k, int skip_gdma, void *id_node, void *id_node_gdma);
typedef void (*nodechip_lrn_fix8b_forward_local)(int bottom_local_offset, int top_local_offset, int imm_buffer_local_offset, int *bottom_dim, float alpha, int size, float beta, float k, int skip_gdma, int shift_fix8b, void *id_node, void *id_node_gdma, int sign_unsign, float scale_in, float scale_out);
typedef void (*nodechip_batchnorm_layer_local)(int bottom_local_offset, int top_local_offset, int mean_local_offset, int variance_local_offset, int Tensor_N, int Tensor_C, int Tensor_H, int Tensor_W, float eps, float scale_ma, int if_relu, float relu_upper_limit, void *id_node);
typedef void (*nodechip_bnscale_fix8b_forward_local)(int bottom_local_offset, int imm_buffer_local_offset, int scale_local_offset, int bias_local_offset, int rshift_local_offset, int top_local_offset, int tensor_n, int tensor_c, int tensor_h, int tensor_w, int bottom_sign, int scale_sign, int bias_sign, int scale_short_str, int bias_short_str, int rshift_short_str, int if_relu, int relu_upper_limit, void *id_node);
typedef void (*nodechip_scale_forward_local)(int bottom_local_offset, int scale_local_offset, int bias_local_offset, int top_local_offset, int *tensor_dim, int *scale_dim, int bias_term, int if_relu, float upper_limit, int is_scale_unaligned, int is_bias_unaligned, void *id_node);
typedef void (*nodechip_eltwise_forward_local)(int *bottom_local_offset, int top_local_offset, int tensor_n, int tensor_c, int tensor_h, int tensor_w, int *bottom_local_cstride, int bottom_cstride_en, int op_code, float *bottom_coeff, int bottom_num, int if_relu, void *id_node);
typedef void (*nodechip_eltwise_fix8b_forward_local)(int *bottom_local_offset, int top_local_offset, int imm_buffer_local_offset, int tensor_n, int tensor_c, int tensor_h, int tensor_w, int *bottom_local_cstride, int bottom_cstride_en, int op_code, int *scale_weight, int *rshift_num, int *bottom_sign, int bottom_num, int if_relu, void *id_node);
typedef void (*nodechip_fc_forward_local)(int bottom_local_offset, int weight_local_offset, int bias_local_offset, int top_local_offset, int slope_local_offset, int imm_buffer_offset, int *bottom_dim, int output_num, int using_bias, int active_type, int channel_shared, float shared_slope, void *id_node);
typedef void (*nodechip_prelu_forward_local_v2)(int bottom_local_offset, int top_local_offset, int slope_local_offset, int buffer_local_offset, int channel_shared, float shared_slope, int *tensor_dim, int st_by_fcway, void *id_node);
typedef void (*nodechip_relu_forward_local)(int bottom_local_offset, int top_local_offset, int *tensor_dim, float upper_limit, void *id_node);
typedef void (*nodechip_prelu_forward_local_fix8b_v3)(int bottom_local_offset, int top_local_offset, int slope_local_offset, int buffer_local_offset, int channel_shared, int shared_slope, uint32_t *tensor_dim, int st_by_fcway, int input_sign, int slope_sign, int output_sign, int rshift_bit, int upper_limit, void *id_node);
typedef void (*nodechip_relu_forward_local_fix16b)(int bottom_local_offset, int top_local_offset, uint32_t *tensor_dim, int input_sign, int output_sign, void *id_node);
typedef void (*nodechip_reorg_forward_local)(int bottom_local_offset, int top_local_offset, int imm_buffer_local_offset, int *bottom_dim, int *top_dim, int skip_gdma, void *id_node, void *id_node_gdma);
typedef void (*nodechip_reorg_forward_fix8b_local)(int bottom_local_offset, int top_local_offset, int imm_buffer_local_offset, int *bottom_dim, int *top_dim, int skip_gdma, void *id_node, void *id_node_gdma);
typedef void (*nodechip_permute_forward_local)(uint32_t bottom_lo, int top_lo, int *bottom_dim, int *top_dim, void *id_node);
typedef void (*nodechip_permute_fix8b_forward_local)(uint32_t bottom_lo, int top_lo, int *bottom_dim, int *top_dim, void *id_node);
typedef void (*nodechip_normalize_forward_local)(int bottom_lo, int top_lo, int imm_buffer_lo, int scale_lo, int *bottom_dim, int across_spatial, int channel_shared, float eps, float scale_val, int if_relu, CMD_ID_NODE *pid_node);
typedef void (*nodechip_normalize_fix8b_forward_local)(int bottom_lo, int top_lo, int imm_buffer_lo, int scale_lo, int *bottom_dim, int across_spatial, int channel_shared, float eps, float scale_val, int if_relu, int bottom_sign, int top_sign, CMD_ID_NODE *pid_node);
typedef void (*nodechip_active_forward_local)(uint32_t bottom_lo, uint32_t top_lo, uint32_t imm_lo, int n, int c, int h, int w, int t, void *param, CMD_ID_NODE *pid_node);
typedef void (*nodechip_mulshift_fix8b_forward_local)(int bottom_local_offset, int imm_buffer_local_offset, int top_local_offset, int tensor_n, int tensor_c, int tensor_h, int tensor_w, int scale_value, int rshift_num, int input_sign, int scale_sign, int output_sign, CMD_ID_NODE *id_node);
typedef void (*nodechip_concat_md)(int concat_axis, int shape_size, int bottom_size, uint64_t *bottom_global_offset, uint64_t top_global_offset, int (*bottton_shape)[8], int *toptensor_shape, int *st_by_concatway, CMD_ID_NODE *id_node);
typedef void (*nodechip_concat_md_fix8b)(int concat_axis, int shape_size, int bottom_size, uint64_t *bottom_global_offset, uint64_t top_global_offset, int (*bottton_shape)[8], int *toptensor_shape, int *st_by_concatway, int in_stmode, int out_stmode, CMD_ID_NODE *id_node);
typedef void (*nodechip_stride_slice_forward_local)(int bottom_local_offset, int top_local_offset, int *input_shape, int input_dim, int begin_mask, int end_mask, int *begin_index, int *end_index, int *stride, CMD_ID_NODE *pid_node);
typedef void (*nodechip_stride_slice_forward_local_fix8b)(int bottom_local_offset, int top_local_offset, int *input_shape, int input_dim, int begin_mask, int end_mask, int *begin_index, int *end_index, int *stride, CMD_ID_NODE *pid_node);
typedef void (*nodechip_pooling3d_local)(int input_offset, int output_offset, int buffer_offset, int n, int c, int it, int ih, int iw, int ot, int oh, int ow, int kt, int kh, int kw, int stride_t, int stride_h, int stride_w, int pad_t, int pad_t_after, int pad_h, int pad_h_after, int pad_w, int pad_w_after, int is_avg_pooling, int avg_pooling_mode, int if_relu, float relu_upper_limit, CMD_ID_NODE *pid_node);
typedef void (*nodechip_conv3d_local)(uint32_t bottom_offset, uint32_t weight_offset, uint32_t bias_offset, uint32_t top_offset, int *input_shape, int *output_shape, int groups, int kt, int kh, int kw, int dt, int dh, int dw, int pad_t, int pad_t_after, int pad_h, int pad_h_after, int pad_w, int pad_w_after, int stride_t, int stride_h, int stride_w, int using_bias, int if_relu, float relu_upper_limit, CMD_ID_NODE *pid_node);
typedef void (*nodechip_conv3d_fix8b_local)(uint32_t bottom_offset, uint32_t weight_offset, uint32_t bias_offset, uint32_t top_offset, int *input_shape, int *output_shape, int kt, int kh, int kw, int dt, int dh, int dw, int pad_t, int pad_t_after, int pad_h, int pad_h_after, int pad_w, int pad_w_after, int stride_t, int stride_h, int stride_w, int using_bias, int if_relu, int in_sign, int weight_sign, int bias_sign, int out_sign, int rshift_num, CMD_ID_NODE *pid_node);
typedef void (*nodechip_deconv3d_local)(uint32_t bottom_offset, uint32_t weight_offset, uint32_t bias_offset, uint32_t top_offset, int *input_shape, int *output_shape, int groups, int kt, int kh, int kw, int dt, int dh, int dw, int pad_t, int pad_t_after, int pad_h, int pad_h_after, int pad_w, int pad_w_after, int stride_t, int stride_h, int stride_w, int using_bias, int if_relu, float relu_upper_limit, CMD_ID_NODE *pid_node);
typedef void (*nodechip_fc_forward_parallel)(uint64_t global_offset_bottom_data, uint64_t global_offset_weight_data, uint64_t global_offset_bias_data, uint64_t global_offset_top_data, uint64_t global_offset_slope_data, int input_row_num, int input_col_num, int weight_col_num, int transpose, int have_bias, int active_type, int channel_shared, float shared_slope, int W_param, CMD_ID_NODE *id_node);
typedef void (*nodechip_fc_fix8b_forward_parallel)(uint64_t bottom_global_addr, uint64_t weight_global_addr, uint64_t bias_global_addr, uint64_t top_global_addr, uint64_t scale_global_addr, int batch_size, int input_neuron_num, int output_neuron_num, int transpose, int have_bias, int bottom_sign, int weight_sign, int bias_sign, int right_shift_bit, int res_16b, int if_relu, int if_global_in_4N, int weight_is_datatensor, int if_global_out_4N, float perlayer_bias, int scale_method, void *quantized_params, CMD_ID_NODE *id_node);
typedef void (*nodechip_pooling_forward_parallel_with_data_split)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, int input_n, int input_c, int input_h, int input_w, int output_h, int output_w, int kh, int kw, int pad_h, int pad_w, int stride_h, int stride_w, int is_avg_pooling, int avg_pooling_mode, int if_relu, CMD_ID_NODE *pid_node);
typedef void (*nodechip_pooling_fix8b_forward_parallel_with_data_split)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, int input_n, int input_c, int input_h, int input_w, int kh, int kw, int pad_h_top, int pad_h_bottom, int pad_w_left, int pad_w_right, int stride_h, int stride_w, int ins0_w, int ins0_h, int is_avg_pooling, int avg_pooling_mode, int r_shift, int using_bias, int rshift_typ, int opd0_sign, int opd1_sign, int opd2_sign, int res0_sign, int if_relu, CMD_ID_NODE *pid_node);
typedef void (*nodechip_pooling_tf_fix8b_forward_parallel_with_data_split)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, int input_n, int input_c, int input_h, int input_w, int kh, int kw, int pad_h_top, int pad_h_bottom, int pad_w_left, int pad_w_right, int stride_h, int stride_w, int ins0_w, int ins0_h, int is_avg_pooling, int r_shift, int using_bias, int rshift_typ, int opd0_sign, int opd1_sign, int opd2_sign, int res0_sign, int if_relu, CMD_ID_NODE *pid_node);
typedef void (*nodechip_sort_per_dim)(uint64_t input_global_addr, uint64_t index_global_addr, uint64_t output_global_addr, uint64_t data_buffer_global_addr, uint64_t index_buffer_global_addr, int *input_shape, int input_dims, int sort_dim, int is_argsort, int stable, int descending, CMD_ID_NODE *pid_node);
typedef void (*nodechip_sort_per_dim_fix8b)(uint64_t input_global_addr, uint64_t index_global_addr, uint64_t output_global_addr, uint64_t data_buffer_global_addr, uint64_t data_buffer2_global_addr, uint64_t index_buffer_global_addr, int *input_shape, int input_dims, int sort_dim, int is_argsort, int stable, int descending, CMD_ID_NODE *pid_node);
typedef void (*nodechip_index_select)(uint64_t input_global_addr, uint64_t index_global_addr, int index_num, uint64_t output_global_addr, uint64_t buffer1_global_addr, uint64_t buffer2_global_addr, int *input_shape, int input_dims, int dim, CMD_ID_NODE *pid_node);
typedef void (*nodechip_index_select_fix8b)(uint64_t input_global_addr, uint64_t index_global_addr, int index_num, uint64_t output_global_addr, uint64_t buffer1_global_addr, uint64_t buffer2_global_addr, int *input_shape, int input_dims, int dim, CMD_ID_NODE *pid_node);
typedef void (*nodechip_psroipooling_forward_with_datasplit)(int nodechip_idx, uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, uint64_t rois_offset_global, uint64_t arm_reserved_global_offset, int input_n, int input_c, int input_h, int input_w, int output_dim, int group_size, int roi_num, float spatial_scale, CMD_ID_NODE *id_node);
typedef void (*nodechip_psroipooling_fix8b_forward_with_datasplit)(int nodechip_idx, uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, uint64_t rois_offset_global, uint64_t arm_reserved_global_offset, int input_n, int input_c, int input_h, int input_w, int output_dim, int group_size, int roi_num, float spatial_scale, int input_sign, int output_sign, CMD_ID_NODE *id_node);
typedef void (*nodechip_roi_pooling_forward)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, uint64_t rois_offset_global, uint64_t arm_reserved_global_offset, int input_n, int input_c, int input_h, int input_w, int pooled_h, int pooled_w, int roi_num, float spatial_scale, CMD_ID_NODE *id_node);
typedef void (*nodechip_crop)(uint64_t bottom_global_offset, uint64_t top_global_offset, int *offset, int *topshape, int *bottomshape, CMD_ID_NODE *id_node);
typedef void (*nodechip_crop_fix8b)(uint64_t bottom_global_offset, uint64_t top_global_offset, int *offset, int *topshape, int *bottomshape, CMD_ID_NODE *id_node);
typedef void (*nodechip_upsample_forward_parallel_with_data_split)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, int input_n, int input_c, int input_h, int input_w, int size, int if_relu, CMD_ID_NODE *pid_node);
typedef void (*nodechip_upsample_forward_parallel_fix8b)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, int input_n, int input_c, int input_h, int input_w, int size, int if_relu, CMD_ID_NODE *id_node);
typedef void (*nodechip_upsample_mask_forward)(uint64_t bottom_global_offset, uint64_t bottom_mask_global_offset, uint64_t top_global_offset, int bottom_global_N, int bottom_c, int bottom_h, int bottom_w, int top_c, int top_h, int top_w, CMD_ID_NODE *pid_node);
typedef void (*nodechip_multiregion_forward_parallel)(uint64_t *bottom_global_offset, uint64_t *top_global_offset, int *Tensor_N, int *Tensor_C, int *Tensor_H, int *Tensor_W, int classes, int coords, int input_num, int nums, int *Activate_param, CMD_ID_NODE *id_node);
typedef void (*nodechip_deconv_forward_parallel_with_data_split)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, uint64_t weight_offset_global, uint64_t bias_offset_global, int input_n, int input_c, int input_h, int input_w, int groups, int output_c, int kh, int kw, int dh, int dw, int pad_h, int pad_h_after, int pad_w, int pad_w_after, int stride_h, int stride_w, int output_padding_h, int output_padding_w, int using_bias, int result_add, int if_relu, int coef_st_way, CMD_ID_NODE *pid_node);
typedef void (*nodechip_deconv_fix16b_forward_parallel)(uint64_t ifmap_global_addr, uint64_t ofmap_global_addr, uint64_t weight_global_addr, uint64_t bias_global_addr, int input_n, int input_c, int input_h, int input_w, int output_c, int groups, int kh, int kw, int dh, int dw, int pad_h_t, int pad_h_b, int pad_w_l, int pad_w_r, int stride_h, int stride_w, int output_padding_h, int output_padding_w, int using_bias, int if_relu, int rshift_num, int coef_st_way, int ifmap_sign, int weight_sign, int bias_sign, CMD_ID_NODE *pid_node);
typedef void (*nodechip_deconv_fix8b_forward_parallel)(uint64_t ifmap_global_addr, uint64_t ofmap_global_addr, uint64_t weight_global_addr, uint64_t bias_global_addr, int input_n, int input_c, int input_h, int input_w, int output_c, int groups, int kh, int kw, int dh, int dw, int pad_h_t, int pad_h_b, int pad_w_l, int pad_w_r, int stride_h, int stride_w, int output_padding_h, int output_padding_w, int using_bias, int if_relu, int rshift_num, int coef_st_way, int ifmap_sign, int weight_sign, int bias_sign, CMD_ID_NODE *pid_node);
typedef void (*nodechip_depthwise_forward_parallel_with_dilation)(uint64_t input_global_offset, uint64_t output_global_offset, uint64_t weight_global_offset, uint64_t bias_global_offset, int input_n, int input_c, int input_h, int input_w, int kernel_h, int kernel_w, int pad_h, int pad_h_after, int pad_w, int pad_w_after, int stride_h, int stride_w, int dilate_h, int dilate_w, int using_bias, int if_relu, float relu_upper_limit, CMD_ID_NODE *pid_node);
typedef void (*nodechip_conv_forward_parallel_with_data_split)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, uint64_t weight_offset_global, uint64_t bias_offset_global, int input_n, int input_c, int input_h, int input_w, int groups, int output_c, int kh, int kw, int dh, int dw, int pad_h, int pad_h_after, int pad_w, int pad_w_after, int stride_h, int stride_w, int using_bias, int result_add, int if_relu, float relu_upper_limit, CMD_ID_NODE *pid_node);
typedef void (*nodechip_winograd_forward_parallel_with_data_split)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, uint64_t weight_offset_global, uint64_t bias_offset_global, int input_n, int input_c, int input_h, int input_w, int groups, int output_c, int pad_h, int pad_h_after, int pad_w, int pad_w_after, int using_bias, int result_add, int if_relu, float relu_upper_limit, int winograd_flag, CMD_ID_NODE *pid_node);
typedef void (*nodechip_winograd_forward_parallel_fix8b_with_data_split)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, uint64_t weight_offset_global, uint64_t bias_offset_global, int input_n, int input_c, int input_h, int input_w, int groups, int output_c, int pad_h, int pad_h_after, int pad_w, int pad_w_after, int using_bias, int if_relu, int upper_limit, int winograd_flag, int ins_h, int ins_w, int rshiftbits, int opd0_sign, int opd1_sign, int opd2_sign, int opd0_short_str, int if_concat_scale, int concat_scale_val, int concat_scale_rshift, int concat_output_sign, CMD_ID_NODE *pid_node);
typedef void (*nodechip_lstm_forward_parallel)(uint64_t x_global_offset, uint64_t cont_global_offset, uint64_t x_static_global_offset, uint64_t h_0_global_offset, uint64_t c_0_global_offset, uint64_t x_weight_global_offset, uint64_t x_bias_global_offset, uint64_t xstatic_weight_global_offset, uint64_t h_weight_global_offset, uint64_t h_global_offset, uint64_t c_T_global_offset, uint64_t h_T_global_offset, uint64_t wxc_buf_global_offset, uint64_t wxc_sta_buf_global_offset, int batch_num, int time_num, int input_dim, int output_dim, int with_x_static, int expose_hidden, CMD_ID_NODE *pid_node);
typedef void (*nodechip_scale_forward)(uint64_t bottom_global_offset, uint64_t scale_global_offset, uint64_t bias_global_offset, uint64_t top_global_offset, int bottom_n, int bottom_c, int bottom_h, int bottom_w, int shape_axis, int shape_axis_num, int add_bias, int if_relu, CMD_ID_NODE *pid_node);
typedef void (*nodechip_eltwise_forward)(uint64_t *bottom_global_offset, uint64_t top_global_offset, uint64_t mask_global_offset, uint64_t buffer_offset, int *input1_shape, int *input2_shape, int input_num, int input_dims, int op_code, float *coeff, int need_mask, float *mask_index, int if_relu, CMD_ID_NODE *pid_node);
typedef void (*nodechip_eltwise_fix8b_forward_parallel)(uint64_t bottom_A_global_addr, uint64_t bottom_B_global_addr, uint64_t top_global_addr, int tensor_n, int tensor_c, int tensor_h, int tensor_w, int op_code, int scale_A, int scale_B, int sign_A, int sign_B, int rshift_A, int rshift_B, int if_relu, CMD_ID_NODE *id_node);
typedef void (*nodechip_prelu_forward)(uint64_t bottom_global_addr, uint64_t slope_global_addr, uint64_t top_global_addr, float slope_val, int channel_shared, int Tensor_N, int Tensor_C, int Tensor_H, int Tensor_W, CMD_ID_NODE *id_node);
typedef void (*nodechip_prelu_forward_fix8b)(uint64_t bottom_global_addr, uint64_t slope_global_addr, uint64_t top_global_addr, float slope_val, int channel_shared, int Tensor_N, int Tensor_C, int Tensor_H, int Tensor_W, int input_sign, int slope_sign, int output_sign, int rshift_bit, int if_global_4N, CMD_ID_NODE *id_node);
typedef void (*nodechip_relu_forward_fix16b)(uint64_t bottom_global_addr, uint64_t top_global_addr, int Tensor_N, int Tensor_C, int Tensor_H, int Tensor_W, int input_sign, int output_sign, int if_global_2N, CMD_ID_NODE *id_node);
typedef void (*nodechip_permute_forward)(uint64_t input_global_offset, uint64_t output_global_offset, int input_n, int input_c, int input_h, int input_w, int *permute_order, CMD_ID_NODE *pid_node);
typedef void (*nodechip_permute_fix8b_forward)(uint64_t input_global_offset, uint64_t output_global_offset, int input_n, int input_c, int input_h, int input_w, int *permute_order, CMD_ID_NODE *pid_node);
typedef void (*nodechip_normalize_forward)(uint64_t bottom_global_offset, uint64_t top_global_offset, uint64_t scale_global_offset, bool across_spatial, bool channel_share, int n, int c, int h, int w, float eps, float scale, int if_relu, CMD_ID_NODE *pid_node);
typedef void (*nodechip_normalize_fix8b_forward)(uint64_t bottom_global_offset, uint64_t top_global_offset, uint64_t scale_global_offset, bool across_spatial, bool channel_share, int n, int c, int h, int w, float eps, float scale, int if_relu, int in_sign, int out_sign, CMD_ID_NODE *pid_node);
typedef void (*nodechip_slice_forward)(uint64_t input_global_offset, int output_tensor_num, uint64_t *output_global_offset, int input_n, int input_c, int input_h, int input_w, int slice_axis, int *output_slice_axis, int *output_slice_en, CMD_ID_NODE *id_node);
typedef void (*nodechip_softmax_forward_parallel)(uint64_t bottom_global_offset, uint64_t top_global_offset, int Tensor_N, int Tensor_C, int Tensor_H, int Tensor_W, int original_n, int original_c, int original_h, int original_w, int in_tensor_global_store_mode, uint64_t global_offset_1N_buf, int bottom_prec, float scale_val, bool log, CMD_ID_NODE *id_node);
typedef void (*nodechip_active_forward_parallel)(uint64_t bottom_global_addr, uint64_t top_global_addr, uint64_t length, int active_type, void *param, CMD_ID_NODE *pid_node);
typedef void (*nodechip_active_forward_parallel_fix8b)(uint64_t bottom_global_addr, uint64_t top_global_addr, uint64_t length, int active_type, float input_scale, float output_scale, int input_signed, int output_signed, void *param, CMD_ID_NODE *pid_node);
typedef void (*nodechip_relu_forward_32bit_parallel)(uint64_t global_offset_bottom_data, uint64_t global_offset_top_data, float negative_slope, float limit, int Tensor_N, int Tensor_C, int Tensor_H, int Tensor_W, CMD_ID_NODE *pid_node);
typedef void (*nodechip_batchnorm_forward_inference_parallel)(uint64_t bottom_global_offset, uint64_t mean_ma_global_offset, uint64_t variance_ma_global_offset, float scale_ma, uint64_t variance_global_offset, uint64_t top_global_offset, float eps, int Tensor_N, int Tensor_C, int Tensor_H, int Tensor_W, int need_var, int need_calc, int if_relu, CMD_ID_NODE *pid_node);
typedef void (*nodechip_batchnorm_forward_inference_parallel_v2)(uint64_t bottom_global_offset, uint64_t mean_ma_global_offset, uint64_t variance_ma_global_offset, float scale_ma, uint64_t variance_global_offset, uint64_t top_global_offset, float eps, int Tensor_N, int Tensor_C, int Tensor_H, int Tensor_W, int need_var, int need_calc, int if_relu, float relu_upper_limit, CMD_ID_NODE *pid_node);
typedef void (*nodechip_bnscale_forward_parallel_fix8b_with_src_storage_mode)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, uint64_t scale_offset_global, uint64_t bias_offset_global, uint64_t rshift_offset_global, int input_n, int input_c, int input_h, int input_w, int input_sign, int scale_sign, int bias_sign, int src_storage_mode, int if_relu, int relu_limit, CMD_ID_NODE *pid_node);
typedef void (*nodechip_lrn_forward_parallel)(uint64_t bottom_uint64_t_offset, uint64_t top_global_offset, int Tensor_N, int Tensor_C, int Tensor_H, int Tensor_W, float alpha, int size, float beta, float k, CMD_ID_NODE *id_node);
typedef void (*nodechip_lrn_fix8b_forward_parallel)(uint64_t bottom_global_offset, uint64_t top_global_offset, int Tensor_N, int Tensor_C, int Tensor_H, int Tensor_W, int sign_unsign, float alpha, int size, float beta, float k, float scale_in, float scale_out, CMD_ID_NODE *id_node);
typedef void (*nodechip_depthwise_fix8b_forward_parallel)(uint64_t input_global_offset, uint64_t output_global_offset, uint64_t weight_global_offset, uint64_t bias_global_offset, int input_n, int input_c, int input_h, int input_w, int kernel_h, int kernel_w, int pad_h_top, int pad_h_bottom, int pad_w_left, int pad_w_right, int stride_h, int stride_w, int ins0_w, int ins0_h, int r_shift, int using_bias, int rshift_typ, int opd0_sign, int opd1_sign, int opd2_sign, int res0_sign, int if_relu, CMD_ID_NODE *pid_node);
typedef void (*nodechip_conv_forward_parallel_fix8b_with_data_split)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, uint64_t weight_offset_global, uint64_t bias_offset_global, int input_n, int input_c, int input_h, int input_w, int groups, int output_c, int kh, int kw, int dh, int dw, int pad_h, int pad_h_after, int pad_w, int pad_w_after, int stride_h, int stride_w, int using_bias, int result_add, int if_relu, int upper_limit, int if_ic_inner, int ins_h, int ins_w, int rshiftbits, int opd0_sign, int opd1_sign, int opd2_sign, int opd0_short_str, int if_concat_scale, int concat_scale_val, int concat_scale_rshift, int concat_output_sign, int use_3ic_optimize, CMD_ID_NODE *pid_node);
typedef void (*nodechip_conv_forward_parallel_fix16b_with_data_split)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, uint64_t weight_offset_global, uint64_t bias_offset_global, int input_n, int input_c, int input_h, int input_w, int groups, int output_c, int kh, int kw, int dh, int dw, int pad_h, int pad_h_after, int pad_w, int pad_w_after, int stride_h, int stride_w, int using_bias, int result_add, int if_relu, int upper_limit, int if_ic_inner, int ins_h, int ins_w, int rshiftbits, int opd0_sign, int opd1_sign, int opd2_sign, int opd0_short_str, CMD_ID_NODE *pid_node);
typedef void (*nodechip_mulshift_fix8b_forward)(uint64_t bottom_local_offset, uint64_t top_local_offset, int tensor_n, int tensor_c, int tensor_h, int tensor_w, int scale_value, int rshift_num, int input_sign, int scale_sign, int output_sign, CMD_ID_NODE *id_node);
typedef void (*nodechip_global_conv_data_split_fix8b)(int input_n, int input_c, int input_h, int input_w, int output_c, int output_h, int output_w, int groups, int kh, int kw, int dh, int dw, int stride_h, int stride_w, int using_bias, int ins_h, int ins_w, int *nsecs, int *hsecs, int *icsecs, int *ocsecs);
typedef void (*nodechip_rpnproposal_forward)(uint64_t global_offset_bottom_data0, uint64_t global_offset_bottom_data1, uint64_t global_offset_bottom_data2, uint64_t global_offset_top_data, uint32_t *bottom0_dim, uint32_t *bottom1_dim, int feat_stride_, int base_size_, int min_size_, int pre_nms_topN_, int post_nms_topN_, float nms_thresh_, float score_thresh_, int in_tensor_global_store_mode_, uint64_t global_offset_1N_buf_, uint64_t arm_reserved_global_offset, int bottom_prec_, float scale_val_, CMD_ID_NODE *id_node);
typedef void (*nodechip_shuffle_channel_forward)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, int input_n, int input_c, int input_h, int input_w, int group_num, CMD_ID_NODE *pid_node);
typedef void (*nodechip_shuffle_channel_fix8b_forward)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, int input_n, int input_c, int input_h, int input_w, int group_num, CMD_ID_NODE *pid_node);
typedef void (*nodechip_topk)(uint64_t bottom_global_offset, uint64_t top_value_global_offset, uint64_t top_index_global_offset, int k, int dim, int Tensor_N, int Tensor_C, int Tensor_H, int Tensor_W, CMD_ID_NODE *pid_node);
typedef void (*nodechip_topk_fix8b)(uint64_t bottom_global_offset, uint64_t top_value_global_offset, uint64_t top_index_global_offset, int k, int dim, int Tensor_N, int Tensor_C, int Tensor_H, int Tensor_W, int in_sign, CMD_ID_NODE *pid_node);
typedef void (*nodechip_lut_v2)(uint64_t bottom_global_offset, uint64_t top_global_offset, uint64_t table_global_offset, int Tensor_N, int Tensor_C, int Tensor_H, int Tensor_W, int bottom_stmode, int top_dtype, int top_stmode, CMD_ID_NODE *pid_node);
typedef void (*nodechip_cumsum)(uint64_t bottom_global_offset, uint64_t top_global_offset, int dim, int Tensor_N, int Tensor_C, int Tensor_H, int Tensor_W, CMD_ID_NODE *pid_node);
typedef void (*nodechip_arg)(uint64_t input_offset_global, uint64_t value_offset_global, uint64_t index_offset_global, int input_n, int input_c, int input_h, int input_w, int axis, int method, int is_index_int32, CMD_ID_NODE *pid_node);
typedef void (*nodechip_arg_local)(uint32_t input_lmem_offset, uint32_t index_lmem_offset, uint32_t argval_lmem_offset, uint32_t imm_lmem_offset, int input_n, int input_c, int input_h, int input_w, int axis, int method, int is_index_int32, CMD_ID_NODE *pid_node, CMD_ID_NODE *pid_node_gdma);
typedef void (*nodechip_arg_fix8b)(uint64_t input_offset_global, uint64_t input_1N_global_addr, uint64_t value_offset_global, uint64_t output_1N_global_addr, uint64_t index_offset_global, uint64_t imm_index_global_addr, int input_sign, int input_n, int input_c, int input_h, int input_w, int axis, int method, CMD_ID_NODE *pid_node);
typedef void (*nodechip_stride_slice_md)(uint64_t input_global_addr, uint64_t output_global_addr, int *input_shape, int shape_size, int begin_mask, int end_mask, int *begin_index, int *end_index, int *stride, CMD_ID_NODE *pid_node);
typedef void (*nodechip_stride_slice_fix8b)(uint64_t input_global_addr, uint64_t output_global_addr, uint64_t buffer_global_addr, uint64_t imm_global_addr, uint64_t *buffer_size, int *input_shape, int shape_size, int in_store_mode, int out_store_mode, int begin_mask, int end_mask, int *begin_index, int *end_index, int *stride, int shrink_axis_mask, CMD_ID_NODE *pid_node);
typedef void (*nodechip_split_tf_md)(uint64_t bottom_global_offset, uint64_t *top_global_offset, int shape_dim, int *shape, int axis, int *split_size, int split_num, CMD_ID_NODE *pid_node);
typedef void (*nodechip_split_tf_fix8b_md)(uint64_t bottom_global_offset, uint64_t buffer_global_addr, uint64_t imm_global_addr, uint64_t *buffer_size, int *input_shape, int shape_size, int in_store_mode, int out_store_mode, int axis, int *split_size, int split_num, uint64_t *top_global_offset, CMD_ID_NODE *pid_node);
typedef void (*nodechip_interp_forward_parallel)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, int input_n, int input_c, int input_h, int input_w, int pad_bag, int pad_end, int output_h, int output_w, bool align_corners, bool half_pixel_centers, int platform_sp, CMD_ID_NODE *id_node);
typedef void (*nodechip_interp_forward_fix8b_parallel)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, int input_n, int input_c, int input_h, int input_w, int pad_bag, int pad_end, int output_h, int output_w, int is_nearest, int bottom_prec, CMD_ID_NODE *id_node);
typedef void (*nodechip_reverse_forward_v2)(uint64_t input_global_offset, uint64_t output_global_offset, const int *shape, int dims, int axis, CMD_ID_NODE *pid_node);
typedef void (*nodechip_reorg_forward_v2)(uint64_t bottom_global_addr, uint64_t top_global_addr, uint64_t buffer_global_addr, int *bottom_shape, int bottom_dims, int stride, int inversed, CMD_ID_NODE *pid_node);
typedef void (*nodechip_reorg_forward_fix8b)(uint64_t bottom_global_offset, uint64_t top_global_offset, int *bottom_dim, int *top_dim, int skip_gdma, CMD_ID_NODE *pid_node);
typedef void (*nodechip_adaptive_pooling_forward)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, int input_n, int input_c, int input_h, int input_w, int pooled_h, int pooled_w, int is_avg_pooling, CMD_ID_NODE *pid_node);
typedef void (*nodechip_yolo)(uint64_t input_offset_global, uint64_t value_offset_global, int input_n, int input_c, int input_h, int input_w, int n, int classes, int coords, int background, int softmax, CMD_ID_NODE *pid_node);
typedef void (*nodechip_memset)(uint64_t global_offset, int input_n, int input_c, int input_h, int input_w, int stride_n, int stride_c, int stride_h, float const_val, CMD_ID_NODE *pid_node);
typedef void (*nodechip_channel_shift_forward)(uint64_t A_global_offset, uint64_t R_global_offset, int n, int c, int h, int w, uint32_t shift_dir, uint32_t shift_num, CMD_ID_NODE *pid_node);
typedef void (*nodechip_channel_shift_forward_fix8b)(uint64_t A_global_offset, uint64_t R_global_offset, int n, int c, int h, int w, uint32_t shift_dir, uint32_t shift_num, CMD_ID_NODE *pid_node);
typedef void (*nodechip_interleave)(uint64_t A_global_offset, uint64_t B_global_offset, uint64_t R_global_offset, int input_n, int input_c, int input_h, int input_w, int axis, int step, CMD_ID_NODE *pid_node);
typedef void (*nodechip_interleave_fix8b)(uint64_t A_global_offset, uint64_t B_global_offset, uint64_t R_global_offset, int input_n, int input_c, int input_h, int input_w, int axis, int step, CMD_ID_NODE *pid_node);
typedef void (*nodechip_interleave_local)(uint32_t bottom0_offset, uint32_t bottom1_offset, uint32_t top_offset, uint32_t *bottom_shape, bool a_is_coeff, bool b_is_coeff, int axis, int step, CMD_ID_NODE *pid_node);
typedef void (*nodechip_interleave_fixpoint_local)(uint32_t bottom0_offset, uint32_t bottom1_offset, uint32_t top_offset, uint32_t *bottom_shape, bool a_is_coeff, bool b_is_coeff, int axis, int step, int is_int8, CMD_ID_NODE *pid_node);
typedef void (*nodechip_stride_calculate_forward)(int nodechip_idx, uint64_t A_global_offset, uint64_t B_global_offset, uint64_t R_global_offset, int input_n, int input_c, int input_h, int input_w, int output_n, int output_c, int output_h, int output_w, int offset_n, int offset_c, int offset_h, int offset_w, int B_N_is_1, int B_C_is_1, int B_H_is_1, int B_W_is_1, int op, int result_add, uint32_t A_is_constant, uint32_t B_is_constant, float A_const_val, float B_const_val, uint32_t stride_n, uint32_t stride_c, uint32_t stride_h, uint32_t stride_w, CMD_ID_NODE *pid_node);
typedef void (*nodechip_stridecalc_forward_global)(uint64_t A_global_offset, uint64_t B_global_offset, uint64_t R_global_offset, uint32_t *bottom0_shape, uint32_t *bottom1_shape, uint32_t *offset, uint32_t *a_stride, uint32_t *b_stride, int shape_dim, int stride_op, bool add_result, bool a_is_const, bool b_is_const, float const_val, CMD_ID_NODE *pid_node);
typedef void (*nodechip_stride_calculate_forward_fix8b)(int nodechip_idx, uint64_t A_global_offset, uint64_t B_global_offset, uint64_t R_global_offset, int input_n, int input_c, int input_h, int input_w, int output_n, int output_c, int output_h, int output_w, int offset_n, int offset_c, int offset_h, int offset_w, int B_N_is_1, int B_C_is_1, int B_H_is_1, int B_W_is_1, int op, int result_add, int input_sign, int output_sign, uint32_t A_is_constant, uint32_t B_is_constant, float A_const_val, float B_const_val, uint32_t stride_n, uint32_t stride_c, uint32_t stride_h, uint32_t stride_w, CMD_ID_NODE *pid_node);
typedef void (*nodechip_stridecalc_forward_local)(uint32_t A_local_offset, uint32_t B_local_offset, uint32_t R_local_offset, uint32_t buffer_offset, uint32_t *bottom0_shape, uint32_t *bottom1_shape, uint32_t *top_shape, uint32_t *offset, uint32_t *a_stride, uint32_t *b_stride, int stride_op, bool add_result, bool a_is_coeff, bool b_is_coeff, bool a_is_const, bool b_is_const, float const_val, CMD_ID_NODE *pid_node);
typedef void (*nodechip_stridecalc_fix8b_forward_global)(uint64_t bottom0_addr, uint64_t bottom1_addr, uint64_t top_addr, uint32_t *bottom0_shape, uint32_t *bottom1_shape, uint32_t *offset, uint32_t *a_stride, uint32_t *b_stride, int shape_dim, int stride_op, bool a_is_const, bool b_is_const, int const_val, int *is_int8, int *is_sign, bool if_relu, bool add_result, CMD_ID_NODE *pid_node);
typedef void (*nodechip_stridecalc_fix8b_forward_local)(uint32_t bottom0_offset, uint32_t bottom1_offset, uint32_t top_offset, uint32_t buffer_offset, uint32_t *bottom0_shape, uint32_t *bottom1_shape, uint32_t *top_shape, uint32_t *offset, uint32_t *a_stride, uint32_t *b_stride, int stride_op, bool a_is_coeff, bool b_is_coeff, bool a_is_const, bool b_is_const, int const_val, int *is_int8, int *is_sign, bool if_relu, bool add_result, CMD_ID_NODE *pid_node);
typedef void (*nodechip_bitwise_forward_global)(uint64_t bottom_A_global_addr, uint64_t bottom_B_global_addr, uint64_t top_global_addr, uint32_t *b0_shape, uint32_t *b1_shape, int num_axes, int is_int8, int if_const, int b_value, int bitwise_op, CMD_ID_NODE *pid_node);
typedef void (*nodechip_bitwise_forward_local)(int bottom_A_local_offset, int bottom_B_local_offset, int top_global_offset, int buffer_local_offset, uint32_t *b0_shape, uint32_t *b1_shape, int num_axes, int is_int8, int if_const, int b_value, int bitwise_op, CMD_ID_NODE *pid_node);
typedef void (*nodechip_arith_shift_forward)(uint64_t bottom_A_global_addr, uint64_t bottom_B_global_addr, uint64_t top_global_addr, int *a_shape, int *b_shape, int a_dim, int b_dim, bool shift_is_const, bool is_num_neuron, int shift_type, int shift_num, int *is_int16, int bottom_sign, CMD_ID_NODE *pid_node);
typedef void (*nodechip_arith_shift_forward_local_v2)(uint32_t bottom_local_offset, uint32_t shift_local_offset, uint32_t top_local_offset, uint32_t buffer_local_offset, int *a_shape, int *b_shape, int a_dim, int b_dim, bool shift_is_const, bool is_num_neuron, int shift_type, int shift_num, int *is_int16, int bottom_sign, int top_sign, CMD_ID_NODE *pid_node);
typedef void (*nodechip_reshape_fix8b)(uint64_t bottom_global_offset, uint64_t top_global_offset, int Tensor_N, int Tensor_C, int Tensor_H, int Tensor_W, int original_n, int original_c, int original_h, int original_w, int in_tensor_global_store_mode, int out_tensor_global_store_mode, CMD_ID_NODE *id_node);
typedef void (*nodechip_pooling3d_forward_parallel)(uint64_t input_addr, uint64_t buffer_addr, uint64_t output_addr, int input_n, int input_c, int input_t, int input_h, int input_w, int output_t, int output_h, int output_w, int kt, int kh, int kw, int pad_t, int pad_t_after, int pad_h, int pad_h_after, int pad_w, int pad_w_after, int stride_t, int stride_h, int stride_w, int is_avg_pooling, int avg_pooling_mode, int if_relu, float relu_upper_limit, CMD_ID_NODE *pid_node);
typedef void (*nodechip_pooling3d_fix8b_forward_parallel)(uint64_t ifmap_offset_global, uint64_t buffer_offset_global, uint64_t ofmap_offset_global, int input_n, int input_c, int input_t, int input_h, int input_w, int kt, int kh, int kw, int pad_t_top, int pad_t_bottom, int pad_h_top, int pad_h_bottom, int pad_w_left, int pad_w_right, int stride_t, int stride_h, int stride_w, int is_avg_pooling, int avg_pooling_mode, int r_shift, int using_bias, int rshift_type, int opd0_sign, int opd1_sign, int opd2_sign, int res0_sign, int if_relu, int relu_upper_limit, CMD_ID_NODE *pid_node);
typedef void (*nodechip_max_pooling_with_mask_forward_v2)(uint64_t bottom_global_offset, uint64_t top_global_offset, uint64_t top_mask_global_offset, int bottom_global_N, int bottom_c, int bottom_h, int bottom_w, int top_h, int top_w, int kh, int kw, int stride_h, int stride_w, int if_relu, float relu_upper_limit, int top_pad_h, int bottom_pad_h, int left_pad_w, int right_pad_w, CMD_ID_NODE *pid_node);
typedef void (*nodechip_conv3d_parallel)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, uint64_t weight_offset_global, uint64_t bias_offset_global, int input_n, int input_c, int input_t, int input_h, int input_w, int groups, int output_c, int kt, int kh, int kw, int dt, int dh, int dw, int pad_t, int pad_t_after, int pad_h, int pad_h_after, int pad_w, int pad_w_after, int stride_t, int stride_h, int stride_w, int using_bias, int if_relu, float relu_upper_limit, int method, CMD_ID_NODE *pid_node);
typedef void (*nodechip_deconv3d)(uint64_t ifmap_offset_global, uint64_t ofmap_offset_global, uint64_t weight_offset_global, uint64_t bias_offset_global, int input_n, int input_c, int input_t, int input_h, int input_w, int groups, int output_c, int kt, int kh, int kw, int dt, int dh, int dw, int pad_t, int pad_t_after, int pad_h, int pad_h_after, int pad_w, int pad_w_after, int stride_t, int stride_h, int stride_w, int output_pad_t, int output_pad_h, int output_pad_w, int using_bias, int if_relu, float relu_upper_limit, CMD_ID_NODE *pid_node);
typedef void (*nodechip_conv3d_fix8b_parallel)(uint64_t input_global_offset, uint64_t output_global_offset, uint64_t filter_global_offset, uint64_t bias_global_offset, int batch, int IC_ALL, int IT, int IH, int IW, int groups, int OC_ALL, int FT, int FH, int FW, int DT, int DH, int DW, int PT0, int PT1, int PH0, int PH1, int PW0, int PW1, int ST, int SH, int SW, int using_bias, int if_relu, int upper_limit, int in_sign, int out_sign, int weight_sign, int bias_sign, int rshift_num, CMD_ID_NODE *pid_node);
typedef void (*nodechip_conv3d_add_parallel)(uint64_t ifmap_offset_global, uint64_t other_offset_global, uint64_t ofmap_offset_global, uint64_t weight_offset_global, uint64_t bias_offset_global, int input_n, int input_c, int input_t, int input_h, int input_w, int groups, int output_c, int kt, int kh, int kw, int dt, int dh, int dw, int pad_t, int pad_t_after, int pad_h, int pad_h_after, int pad_w, int pad_w_after, int stride_t, int stride_h, int stride_w, int using_bias, int if_relu, float relu_upper_limit, int method, CMD_ID_NODE *pid_node);
typedef void (*nodechip_unfold)(uint64_t input_global_offset, int input_rank, int *input_shape, int unfold_axis, int unfold_size, int unfold_step, uint64_t buffer_global_offset, uint64_t *buffer_size, uint64_t output_global_offset, CMD_ID_NODE *id_node);
typedef void (*nodechip_gru)(uint64_t xGlobalAddr, uint64_t h0GlobalAddr, uint64_t yGlobalAddr, uint64_t hnGlobalAddr, uint64_t wGlobalAddr, uint64_t bGlobalAddr, uint64_t zGlobalAddr, bool bias, int sequence, int batch, int xSize, int hSize, bool batchFirst, bool bidirectional, int numLayers, CMD_ID_NODE *id_node);
typedef void (*nodechip_pytorch_lstm)(uint64_t xGlobalAddr, uint64_t h0GlobalAddr, uint64_t c0GlobalAddr, uint64_t yGlobalAddr, uint64_t hnGlobalAddr, uint64_t cnGlobalAddr, uint64_t wGlobalAddr, uint64_t bGlobalAddr, uint64_t zGlobalAddr, bool bias, int sequence, int batch, int xSize, int hSize, bool batchFirst, bool bidirectional, int numLayers, CMD_ID_NODE *id_node);
typedef void (*nodechip_matrix_band_part)(uint64_t input_global_offset, uint64_t output_global_offset, const int *shape, int dim, int lower, int upper, CMD_ID_NODE *id_node);
typedef void (*nodechip_global_memcpy_ex)(uint64_t src_addr, uint64_t dst_addr, int block_num, int src_stride, int dst_stride, int src_dtype, int dst_dtype, int block_size, CMD_ID_NODE *pid_node);
typedef void (*nodechip_lut_local_v2)(int index_local_offset, int table_l2_offset, int imm_local_offset, int output_local_offset, int *index_shape, int index_dim, int bottom_stmode, int top_dtype, int top_stmode, void *pid_node);
typedef void (*nodechip_serial_number_gen)(uint64_t global_addr, uint32_t local_addr, uint32_t local_buffer, int save_in_global, int save_as_float, int C, int HW, void *pid_node);
typedef void (*nodechip_pad_fix8b)(uint64_t bottom_global_offset, uint64_t top_global_offset, bool input_is_4N, uint64_t input_1N_global_offset, bool output_is_4N, uint64_t output_1N_global_offset, int bottom_n, int bottom_c, int bottom_h, int bottom_w, int (*paddings)[2], char pad_val, int pad_op, void *pid_node);
typedef void (*nodechip_pad)(uint64_t bottom_global_offset, uint64_t top_global_offset, int bottom_n, int bottom_c, int bottom_h, int bottom_w, int (*paddings)[2], float pad_val, int pad_op, void *pid_node);
typedef void (*nodechip_pad_local)(int input_local_offset, int output_local_offset, int input_n, int input_c, int input_h, int input_w, int (*pad)[2], int type, float constant, void *pid_node);
typedef void (*nodechip_pad_fix8b_local)(int input_local_offset, int output_local_offset, int input_n, int input_c, int input_h, int input_w, int (*pad)[2], int type, unsigned char constant, void *pid_node);
typedef void (*nodechip_concat_local_v2)(uint32_t *p_bottom_local_offset, uint32_t top_local_offset, int **bottom_dim, int bottom_num, int *is_st_concat_way, int *top_dim, int concat_axis, void *id_node, void *id_node_gdma);
typedef void (*nodechip_concat_fix8b_local_v2)(uint32_t *p_bottom_local_offset, uint32_t top_local_offset, int **bottom_dim, int bottom_num, int *is_st_concat_way, int *top_dim, int concat_axis, void *id_node, void *id_node_gdma);
typedef void (*nodechip_const_binary)(uint64_t bottom_global_addr, uint64_t length, float bottom_val, uint64_t top_global_addr, int binary_op, int inversed, int if_relu, float relu_limit, CMD_ID_NODE *pid_node, int input_is_int32);
typedef void (*nodechip_global_int2float)(uint64_t bottom_global_offset, uint64_t top_global_offset, int input_n, int input_c, int input_h, int input_w, int sign_unsign, TENSOR_STORAGE_MODE mode, CMD_ID_NODE* id_node);
typedef void (*nodechip_float2int8_v2)(uint64_t A_global_offset, uint64_t R_global_offset, int input_n, int input_c, int input_h, int input_w, int sign_unsign, TENSOR_STORAGE_MODE stmode, ROUND_MODE_T round_mode, CMD_ID_NODE* id_node);
typedef void (*nodechip_const_binary_local)(uint32_t bottom0_lo, uint32_t *bottom0_shape, float bottom1_val, uint32_t top_lo, int binary_op, int inversed, int if_relu, float relu_limit, CMD_ID_NODE *pid_node);

// clang-format on
namespace tpu_mlir {
namespace backend {
class BM1684 : public BM168x {
public:
  static BM1684 &instance() {
    static BM1684 inst;
    return inst;
  }

  // -------------------------------------------------------------------
  // functions from nodechip
  // -------------------------------------------------------------------
  // clang-format off
  allow_store_cmd dl_allow_store_cmd;
  forbid_store_cmd dl_forbid_store_cmd;
  tensor_align_move_gen_cmd dl_tensor_align_move_gen_cmd;
  general_matrix_move_gen_cmd dl_general_matrix_move_gen_cmd;
  nodechip_conv_forward_local dl_nodechip_conv_forward_local;
  nodechip_winograd_forward_local dl_nodechip_winograd_forward_local;
  nodechip_pooling_fix8b_forward_local dl_nodechip_pooling_fix8b_forward_local;
  nodechip_conv_forward_local_fix8b dl_nodechip_conv_forward_local_fix8b;
  nodechip_conv_forward_local_fix16b dl_nodechip_conv_forward_local_fix16b;
  nodechip_winograd_forward_local_fix8b dl_nodechip_winograd_forward_local_fix8b;
  nodechip_winograd_double_buffer_forward_local_fix8b dl_nodechip_winograd_double_buffer_forward_local_fix8b;
  nodechip_deconv_forward_local dl_nodechip_deconv_forward_local;
  nodechip_deconv_fix16b_forward_local dl_nodechip_deconv_fix16b_forward_local;
  nodechip_deconv_fix8b_forward_local dl_nodechip_deconv_fix8b_forward_local;
  nodechip_pooling_forward_local dl_nodechip_pooling_forward_local;
  nodechip_upsample_forward_local dl_nodechip_upsample_forward_local;
  nodechip_upsample_fix8b_forward_local dl_nodechip_upsample_fix8b_forward_local;
  nodechip_lrn_forward_local dl_nodechip_lrn_forward_local;
  nodechip_lrn_fix8b_forward_local dl_nodechip_lrn_fix8b_forward_local;
  nodechip_batchnorm_layer_local dl_nodechip_batchnorm_layer_local;
  nodechip_bnscale_fix8b_forward_local dl_nodechip_bnscale_fix8b_forward_local;
  nodechip_scale_forward_local dl_nodechip_scale_forward_local;
  nodechip_eltwise_forward_local dl_nodechip_eltwise_forward_local;
  nodechip_eltwise_fix8b_forward_local dl_nodechip_eltwise_fix8b_forward_local;
  nodechip_fc_forward_local dl_nodechip_fc_forward_local;
  nodechip_prelu_forward_local_v2 dl_nodechip_prelu_forward_local_v2;
  nodechip_relu_forward_local dl_nodechip_relu_forward_local;
  nodechip_prelu_forward_local_fix8b_v3 dl_nodechip_prelu_forward_local_fix8b_v3;
  nodechip_relu_forward_local_fix16b dl_nodechip_relu_forward_local_fix16b;
  nodechip_reorg_forward_local dl_nodechip_reorg_forward_local;
  nodechip_reorg_forward_fix8b_local dl_nodechip_reorg_forward_fix8b_local;
  nodechip_permute_forward_local dl_nodechip_permute_forward_local;
  nodechip_permute_fix8b_forward_local dl_nodechip_permute_fix8b_forward_local;
  nodechip_normalize_forward_local dl_nodechip_normalize_forward_local;
  nodechip_normalize_fix8b_forward_local dl_nodechip_normalize_fix8b_forward_local;
  nodechip_active_forward_local dl_nodechip_active_forward_local;
  nodechip_mulshift_fix8b_forward_local dl_nodechip_mulshift_fix8b_forward_local;
  nodechip_concat_md dl_nodechip_concat_md;
  nodechip_concat_md_fix8b dl_nodechip_concat_md_fix8b;
  nodechip_stride_slice_forward_local dl_nodechip_stride_slice_forward_local;
  nodechip_stride_slice_forward_local_fix8b dl_nodechip_stride_slice_forward_local_fix8b;
  nodechip_pooling3d_local dl_nodechip_pooling3d_local;
  nodechip_conv3d_local dl_nodechip_conv3d_local;
  nodechip_conv3d_fix8b_local dl_nodechip_conv3d_fix8b_local;
  nodechip_deconv3d_local dl_nodechip_deconv3d_local;
  nodechip_fc_forward_parallel dl_nodechip_fc_forward_parallel;
  nodechip_fc_fix8b_forward_parallel dl_nodechip_fc_fix8b_forward_parallel;
  nodechip_pooling_forward_parallel_with_data_split dl_nodechip_pooling_forward_parallel_with_data_split;
  nodechip_pooling_fix8b_forward_parallel_with_data_split dl_nodechip_pooling_fix8b_forward_parallel_with_data_split;
  nodechip_pooling_tf_fix8b_forward_parallel_with_data_split dl_nodechip_pooling_tf_fix8b_forward_parallel_with_data_split;
  nodechip_sort_per_dim dl_nodechip_sort_per_dim;
  nodechip_sort_per_dim_fix8b dl_nodechip_sort_per_dim_fix8b;
  nodechip_index_select dl_nodechip_index_select;
  nodechip_index_select_fix8b dl_nodechip_index_select_fix8b;
  nodechip_psroipooling_forward_with_datasplit dl_nodechip_psroipooling_forward_with_datasplit;
  nodechip_psroipooling_fix8b_forward_with_datasplit dl_nodechip_psroipooling_fix8b_forward_with_datasplit;
  nodechip_roi_pooling_forward dl_nodechip_roi_pooling_forward;
  nodechip_crop dl_nodechip_crop;
  nodechip_crop_fix8b dl_nodechip_crop_fix8b;
  nodechip_upsample_forward_parallel_with_data_split dl_nodechip_upsample_forward_parallel_with_data_split;
  nodechip_upsample_forward_parallel_fix8b dl_nodechip_upsample_forward_parallel_fix8b;
  nodechip_upsample_mask_forward dl_nodechip_upsample_mask_forward;
  nodechip_multiregion_forward_parallel dl_nodechip_multiregion_forward_parallel;
  nodechip_deconv_forward_parallel_with_data_split dl_nodechip_deconv_forward_parallel_with_data_split;
  nodechip_deconv_fix16b_forward_parallel dl_nodechip_deconv_fix16b_forward_parallel;
  nodechip_deconv_fix8b_forward_parallel dl_nodechip_deconv_fix8b_forward_parallel;
  nodechip_depthwise_forward_parallel_with_dilation dl_nodechip_depthwise_forward_parallel_with_dilation;
  nodechip_conv_forward_parallel_with_data_split dl_nodechip_conv_forward_parallel_with_data_split;
  nodechip_winograd_forward_parallel_with_data_split dl_nodechip_winograd_forward_parallel_with_data_split;
  nodechip_winograd_forward_parallel_fix8b_with_data_split dl_nodechip_winograd_forward_parallel_fix8b_with_data_split;
  nodechip_lstm_forward_parallel dl_nodechip_lstm_forward_parallel;
  nodechip_scale_forward dl_nodechip_scale_forward;
  nodechip_eltwise_forward dl_nodechip_eltwise_forward;
  nodechip_eltwise_fix8b_forward_parallel dl_nodechip_eltwise_fix8b_forward_parallel;
  nodechip_prelu_forward dl_nodechip_prelu_forward;
  nodechip_prelu_forward_fix8b dl_nodechip_prelu_forward_fix8b;
  nodechip_relu_forward_fix16b dl_nodechip_relu_forward_fix16b;
  nodechip_permute_forward dl_nodechip_permute_forward;
  nodechip_permute_fix8b_forward dl_nodechip_permute_fix8b_forward;
  nodechip_normalize_forward dl_nodechip_normalize_forward;
  nodechip_normalize_fix8b_forward dl_nodechip_normalize_fix8b_forward;
  nodechip_slice_forward dl_nodechip_slice_forward;
  nodechip_softmax_forward_parallel dl_nodechip_softmax_forward_parallel;
  nodechip_active_forward_parallel dl_nodechip_active_forward_parallel;
  nodechip_active_forward_parallel_fix8b dl_nodechip_active_forward_parallel_fix8b;
  nodechip_relu_forward_32bit_parallel dl_nodechip_relu_forward_32bit_parallel;
  nodechip_batchnorm_forward_inference_parallel dl_nodechip_batchnorm_forward_inference_parallel;
  nodechip_batchnorm_forward_inference_parallel_v2 dl_nodechip_batchnorm_forward_inference_parallel_v2;
  nodechip_bnscale_forward_parallel_fix8b_with_src_storage_mode dl_nodechip_bnscale_forward_parallel_fix8b_with_src_storage_mode;
  nodechip_lrn_forward_parallel dl_nodechip_lrn_forward_parallel;
  nodechip_lrn_fix8b_forward_parallel dl_nodechip_lrn_fix8b_forward_parallel;
  nodechip_depthwise_fix8b_forward_parallel dl_nodechip_depthwise_fix8b_forward_parallel;
  nodechip_conv_forward_parallel_fix8b_with_data_split dl_nodechip_conv_forward_parallel_fix8b_with_data_split;
  nodechip_conv_forward_parallel_fix16b_with_data_split dl_nodechip_conv_forward_parallel_fix16b_with_data_split;
  nodechip_mulshift_fix8b_forward dl_nodechip_mulshift_fix8b_forward;
  nodechip_global_conv_data_split_fix8b dl_nodechip_global_conv_data_split_fix8b;
  nodechip_rpnproposal_forward dl_nodechip_rpnproposal_forward;
  nodechip_shuffle_channel_forward dl_nodechip_shuffle_channel_forward;
  nodechip_shuffle_channel_fix8b_forward dl_nodechip_shuffle_channel_fix8b_forward;
  nodechip_topk dl_nodechip_topk;
  nodechip_topk_fix8b dl_nodechip_topk_fix8b;
  nodechip_lut_v2 dl_nodechip_lut_v2;
  nodechip_cumsum dl_nodechip_cumsum;
  nodechip_arg dl_nodechip_arg;
  nodechip_arg_local dl_nodechip_arg_local;
  nodechip_arg_fix8b dl_nodechip_arg_fix8b;
  nodechip_stride_slice_md dl_nodechip_stride_slice_md;
  nodechip_stride_slice_fix8b dl_nodechip_stride_slice_fix8b;
  nodechip_split_tf_md dl_nodechip_split_tf_md;
  nodechip_split_tf_fix8b_md dl_nodechip_split_tf_fix8b_md;
  nodechip_interp_forward_parallel dl_nodechip_interp_forward_parallel;
  nodechip_interp_forward_fix8b_parallel dl_nodechip_interp_forward_fix8b_parallel;
  nodechip_reverse_forward_v2 dl_nodechip_reverse_forward_v2;
  nodechip_reorg_forward_v2 dl_nodechip_reorg_forward_v2;
  nodechip_reorg_forward_fix8b dl_nodechip_reorg_forward_fix8b;
  nodechip_adaptive_pooling_forward dl_nodechip_adaptive_pooling_forward;
  nodechip_yolo dl_nodechip_yolo;
  nodechip_memset dl_nodechip_memset;
  nodechip_channel_shift_forward dl_nodechip_channel_shift_forward;
  nodechip_channel_shift_forward_fix8b dl_nodechip_channel_shift_forward_fix8b;
  nodechip_interleave dl_nodechip_interleave;
  nodechip_interleave_fix8b dl_nodechip_interleave_fix8b;
  nodechip_interleave_local dl_nodechip_interleave_local;
  nodechip_interleave_fixpoint_local dl_nodechip_interleave_fixpoint_local;
  nodechip_stride_calculate_forward dl_nodechip_stride_calculate_forward;
  nodechip_stridecalc_forward_global dl_nodechip_stridecalc_forward_global;
  nodechip_stride_calculate_forward_fix8b dl_nodechip_stride_calculate_forward_fix8b;
  nodechip_stridecalc_forward_local dl_nodechip_stridecalc_forward_local;
  nodechip_stridecalc_fix8b_forward_global dl_nodechip_stridecalc_fix8b_forward_global;
  nodechip_stridecalc_fix8b_forward_local dl_nodechip_stridecalc_fix8b_forward_local;
  nodechip_bitwise_forward_global dl_nodechip_bitwise_forward_global;
  nodechip_bitwise_forward_local dl_nodechip_bitwise_forward_local;
  nodechip_arith_shift_forward dl_nodechip_arith_shift_forward;
  nodechip_arith_shift_forward_local_v2 dl_nodechip_arith_shift_forward_local_v2;
  nodechip_reshape_fix8b dl_nodechip_reshape_fix8b;
  nodechip_pooling3d_forward_parallel dl_nodechip_pooling3d_forward_parallel;
  nodechip_pooling3d_fix8b_forward_parallel dl_nodechip_pooling3d_fix8b_forward_parallel;
  nodechip_max_pooling_with_mask_forward_v2 dl_nodechip_max_pooling_with_mask_forward_v2;
  nodechip_conv3d_parallel dl_nodechip_conv3d_parallel;
  nodechip_deconv3d dl_nodechip_deconv3d;
  nodechip_conv3d_fix8b_parallel dl_nodechip_conv3d_fix8b_parallel;
  nodechip_conv3d_add_parallel dl_nodechip_conv3d_add_parallel;
  nodechip_unfold dl_nodechip_unfold;
  nodechip_gru dl_nodechip_gru;
  nodechip_pytorch_lstm dl_nodechip_pytorch_lstm;
  nodechip_matrix_band_part dl_nodechip_matrix_band_part;
  nodechip_global_memcpy_ex dl_nodechip_global_memcpy_ex;
  nodechip_lut_local_v2 dl_nodechip_lut_local_v2;
  nodechip_serial_number_gen dl_nodechip_serial_number_gen;
  nodechip_pad_fix8b dl_nodechip_pad_fix8b;
  nodechip_pad dl_nodechip_pad;
  nodechip_pad_local dl_nodechip_pad_local;
  nodechip_pad_fix8b_local dl_nodechip_pad_fix8b_local;
  nodechip_concat_local_v2 dl_nodechip_concat_local_v2;
  nodechip_concat_fix8b_local_v2 dl_nodechip_concat_fix8b_local_v2;
  nodechip_const_binary dl_nodechip_const_binary;
  nodechip_global_int2float dl_nodechip_global_int2float;
  nodechip_float2int8_v2 dl_nodechip_float2int8_v2;
  nodechip_const_binary_local dl_nodechip_const_binary_local;
  // clang-format on
public:
  virtual uint64_t get_gmem_start() override;
  virtual uint64_t get_ctx_start_addr() override;
  virtual uint32_t get_bdc_len(int bdc_num, int group_id) override;
  virtual uint32_t get_gdma_len(int gdma_num, int group_id) override;
  virtual int64_t get_npu_num() override { return 64; }
  virtual int64_t get_eu_bytes() override { return 128; }
  virtual int64_t get_lmem_bytes() override { return (1 << 19); } // 512KB
  virtual int64_t get_lmem_banks() override { return 8; }
  virtual int64_t get_n_align(int64_t dtype_bytes) override {
    // for 4N mode
    return 4 / dtype_bytes;
  }

public:
  static const int64_t NPU_NUM = 64;
  static const int64_t EU_BYTES = 128;
  static const int64_t LMEM_BYTES = 1 << 19; // 512KB
  static const int64_t LMEM_BANKS = 8;
  static const int64_t LMEM_BANK_BYTES = LMEM_BYTES / LMEM_BANKS;
  static const int64_t BDC_CMD_ALIGNED_BIT = 7;
  static const int64_t BDC_CMD_ALIGNED_NUM =
      (1 << BDC_CMD_ALIGNED_BIT) / sizeof(uint32_t);
  static const int64_t GDMA_CMD_ALIGNED_BIT = 7;
  static const int64_t GDMA_CMD_ALIGNED_NUM =
      (1 << GDMA_CMD_ALIGNED_BIT) / sizeof(uint32_t);
  static constexpr llvm::StringRef LIB_NAME = "libbackend_1684.so";

protected:
  BM1684();
  ~BM1684(){};
  template <typename FPtrTy> FPtrTy CastToFPtr(const char *symbolName);
  virtual const char *get_lib_name() override { return LIB_NAME.data(); };
  virtual void load_functions() override;
};
} // namespace backend
} // namespace tpu_mlir

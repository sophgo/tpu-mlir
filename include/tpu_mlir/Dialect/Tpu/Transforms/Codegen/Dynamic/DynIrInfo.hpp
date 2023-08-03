//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <vector>
#include "string.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynCompileCommon.hpp"
namespace tpu_mlir {
namespace tpu {
using std::vector;

typedef struct dynamic_tensor_info {
  u32 max_hslice;
  u32 global_stride_h;
  u32 global_kh;
  u32 global_up_pad_h;
  u32 global_down_pad_h;
} dynamic_tensor_info_t;

#define TENSOR_INFO_INIT_VALUE                                                 \
  { 0, 0, 0, 0, 0, 0, 0 }
typedef struct ir_tensor_info {
  u8 is_io_tensor;
  u8 tensor_type;
  u32 tensor_id;
  u32 local_mem_offset;
  u64 global_mem_offset;
  float coeff;
  u32 consumer_number;
} ir_tensor_info_t;

// IR layer information
typedef struct ir_layer_info {
  ir_layer_info() {
    extra_len = 0;
    extra_buffer = nullptr;
  }
  // return buffer_ptr
  u8 *set_extra_buffer(u32 len, u8 version = 0, const void *buffer = nullptr) {
    if (extra_buffer) {
      delete[] extra_buffer;
    }
    extra_buffer = new u8[len];
    if (buffer != nullptr) {
      memcpy(extra_buffer, buffer, len);
    }
    extra_len = len;
    extra_version = version;
    return extra_buffer;
  }

  bool swpipl_enable;
  u32 stage_and_ir_size;
  int layer_id;
  FW_LAYER_TYPE_T fw_layer_type;
  bool is_cpu_layer;
  DATA_SIZE_T data_size;
  u8 intensor_store_mode;
  u8 outtensor_store_mode;
  u32 layer_fw_ir_length;
  u8 extra_version;
  u32 extra_len;
  u8 *extra_buffer;
  union {
    fw_conv_layer_param_t fw_conv_layer_param;
    fw_rpn_layer_param_t fw_rpn_layer_param;
    fw_fc_layer_param_t fw_fc_layer_param;
    fw_pool_layer_param_t fw_pool_layer_param;
    fw_pooltf_layer_param_t fw_pooltf_layer_param;
    fw_split_tf_layer_param_t fw_split_tf_layer_param;
    // fw_relu_layer_param_t        fw_relu_layer_param;
    fw_prelu_layer_param_t fw_prelu_layer_param;
    fw_softmax_layer_param_t fw_softmax_layer_param;
    fw_lrn_layer_param_t fw_lrn_layer_param;
    fw_eltwise_layer_param_t fw_eltwise_layer_param;
    fw_scale_layer_param_t fw_scale_layer_param;
    fw_batchnorm_layer_param_t fw_batchnorm_layer_param;
    fw_reshape_layer_param_t fw_reshape_layer_param;
    fw_deconv_layer_param_t fw_deconv_layer_param;
    fw_deconv3d_layer_param_t fw_deconv3d_layer_param;
    fw_crop_layer_param_t fw_crop_layer_param;
    fw_concat_loc_layer_param_t fw_concat_loc_layer_param;
    fw_mulshift_layer_param_t fw_mulshift_layer_param;
    fw_pad_layer_param_t fw_pad_layer_param;
    fw_arg_layer_param_t fw_arg_layer_param;
    fw_permute_layer_param_t fw_permute_layer_param;
    fw_tile_layer_param_t fw_tile_layer_param;
    fw_reduce_layer_param_t fw_reduce_layer_param;
    fw_select_layer_param_t fw_select_layer_param;
    fw_where_layer_param_t fw_where_layer_param;
    fw_masked_select_layer_param_t fw_masked_select_layer_param;
    fw_index_select_layer_param_t fw_index_select_layer_param;
    fw_sort_per_dim_layer_param_t fw_sort_per_dim_layer_param;
    fw_nms_layer_param_t fw_nms_layer_param;
    fw_broadcast_binary_layer_param_t fw_broadcast_binary_layer_param;
    fw_eltwise_binary_layer_param_t fw_eltwise_binary_layer_param;
    fw_const_binary_layer_param_t fw_const_binary_layer_param;
    fw_biasadd_layer_param_t fw_biasadd_layer_param;
    fw_active_layer_param_t fw_active_layer_param;
    fw_normalize_layer_param_t fw_normalize_layer_param;
    fw_shape_const_layer_param_t fw_shape_const_layer_param;
    fw_shape_op_layer_param_t fw_shape_op_layer_param;
    fw_shape_slice_layer_param_t fw_shape_slice_layer_param;
    fw_shape_reorder_layer_param_t fw_shape_reorder_layer_param;
    fw_expand_dims_layer_param_t fw_expand_dims_layer_param;
    fw_squeeze_layer_param_t fw_squeeze_layer_param;
    fw_ref_pad_layer_param_t fw_ref_pad_layer_param;
    fw_transpose_layer_param_t fw_transpose_layer_param;
    fw_shape_pack_layer_param_t fw_shape_pack_layer_param;
    fw_reduce_full_layer_param_t fw_reduce_full_layer_param;
    fw_concat_layer_param_t fw_concat_layer_param;
    fw_stride_slice_layer_param_t fw_stride_slice_layer_param;
    fw_space2batch_layer_param_t fw_space2batch_layer_param;
    fw_batch2space_layer_param_t fw_batch2space_layer_param;
    fw_interp_layer_param_t fw_interp_layer_param;
    fw_expand_layer_param_t fw_expand_layer_param;
    fw_embedding_layer_param_t fw_embedding_layer_param;
    fw_topk_layer_param_t fw_topk_layer_param;
    fw_cumsum_layer_param_t fw_cumsum_layer_param;
    fw_shape_addn_layer_param_t fw_shape_addn_layer_param;
    fw_roi_pooling_layer_param_t fw_roi_pooling_layer_param;
    fw_psroi_pooling_layer_param_t fw_psroi_pooling_layer_param;
    fw_constant_fill_layer_param_t fw_constant_fill_layer_param;
    fw_simple_crop_layer_param_t fw_simple_crop_layer_param;
    fw_slicelike_layer_param_t fw_slicelike_layer_param;
    fw_adaptive_pool_layer_param_t fw_adaptive_pool_layer_param;
    fw_batch_matmul_layer_param_t fw_batch_matmul_layer_param;
    fw_upsample_layer_param_t fw_upsample_layer_param;
    fw_shape_tile_layer_param_t fw_shape_tile_layer_param;
    fw_shape_reverse_layer_param_t fw_shape_reverse_layer_param;
    fw_shape_expand_ndims_layer_param_t fw_shape_expand_ndims_layer_param;
    fw_shape_cast_layer_param_t fw_shape_cast_layer_param;
    fw_shape_reduce_layer_param_t fw_shape_reduce_layer_param;
    fw_dtype_convert_layer_param_t fw_dtype_convert_layer_param;
    fw_priorbox_layer_param_t fw_priorbox_layer_param;
    fw_yolo_layer_param_t fw_yolo_layer_param;
    fw_yolov3_detect_out_layer_param_t fw_yolov3_detect_out_layer_param;
    fw_yolov5_detect_out_layer_param_t fw_yolov5_detect_out_layer_param;
    fw_yolov5_decode_detect_out_layer_param_t fw_yolov5_detect_decode_out_layer_param;
    fw_yolov8_detect_out_layer_param_t fw_yolov8_detect_out_layer_param;
    fw_reorg_layer_param_t fw_reorg_layer_param;
    fw_ssd_detect_out_layer_param_t fw_ssd_detect_out_layer_param;
    fw_tensorarray_layer_param_t fw_tensorarray_layer_param;
    fw_shape_split_layer_param_t fw_shape_split_layer_param;
    fw_shape_unary_layer_param_t fw_shape_unary_layer_param;
    fw_shape_squeeze_layer_param_t fw_shape_squeeze_layer_param;
    fw_tensor_array_op_param_t fw_tensor_array_op_param;
    fw_shape_ref_layer_param_t fw_shape_ref_layer_param;
    fw_rank_layer_param_t fw_rank_layer_param;
    fw_coeff2neuron_layer_param_t fw_coeff2neuron_layer_param;
    fw_shape_select_layer_param_t fw_shape_select_layer_param;
    fw_depth2space_layer_param_t fw_depth2space_layer_param;
    fw_where_squeeze_gather_layer_param_t fw_where_squeeze_gather_layer_param;
    fw_reverse_layer_param_t fw_reverse_layer_param;
    fw_lstm_layer_param_t fw_lstm_layer_param;
    fw_broadcast_like_layer_param_t fw_broadcast_like_layer_param;
    fw_lut_layer_param_t fw_lut_layer_param;
    fw_matrix_band_part_layer_param_t fw_matrix_band_part_layer_param;
    fw_slice_layer_param_t fw_slice_layer_param;
    fw_shape_sizeslice_layer_param_t fw_shape_sizeslice_layer_param;
    fw_conv3d_layer_param_t fw_conv3d_layer_param;
    fw_pool3d_layer_param_t fw_pool3d_layer_param;
    fw_stridecalc_layer_param_t fw_stridecalc_layer_param;
    fw_interleave_layer_param_t fw_interleave_layer_param;
    fw_bitwise_layer_param_t fw_bitwise_layer_param;
    fw_binary_shift_layer_param_t fw_binary_shift_layer_param;
    // global_dynamic step 0: add layer param to union part
    fw_arith_shift_layer_param_t fw_arith_shift_layer_param;
    fw_gru_layer_param_t fw_gru_layer_param;
    fw_pytorch_lstm_layer_param_t fw_pytorch_lstm_layer_param;
    fw_multi_masked_select_layer_param_t fw_multi_masked_select_layer_param;
    fw_tpu_layer_param_t fw_tpu_layer_param;
    fw_upsample_mask_layer_param_t fw_upsample_mask_layer_param;
    fw_group_norm_layer_param_t fw_group_norm_layer_param;
  } fw_layer_param_u;

  vector<ir_tensor_info_t> ir_tensor_info_v; // layer input and output

} ir_layer_info_t;

// IR tensor gdma information
typedef struct ir_tensor_gdma_info {
  bool swpipl_enable;
  u32 stage_and_ir_size;
  tensor_gdma_type_t fw_tensor_gdma_type;
  fw_dynamic_output_info_t fw_shape_info;

  union {
    fw_gdma_ld_in_neuron_t fw_gdma_ld_in_neuron;
    fw_gdma_st_out_neuron_t fw_gdma_st_out_neuron;
    fw_gdma_ld_itm_neuron_t fw_gdma_ld_itm_neuron;
    fw_gdma_st_itm_neuron_t fw_gdma_st_itm_neuron;
    fw_gdma_coeff_t fw_gdma_coeff;
    fw_gdma_coeff_neuron_t fw_gdma_coeff_neuron;
    fw_gdma_mv_itm_neuron_t fw_gdma_mv_itm_neuron;
    fw_gdma_mv_out_neuron_t fw_gdma_mv_out_neuron;
    fw_gdma_ld_itm_extend_neuron_t fw_gdma_ld_itm_extend_neuron;
    fw_gdma_st_itm_extend_neuron_t fw_gdma_st_itm_extend_neuron;
    fw_gdma_mv_itm_extend_neuron_t fw_gdma_mv_itm_extend_neuron;
    fw_gdma_ld_g2l2_t fw_gdma_ld_g2l2;
    fw_gdma_st_out_extend_neuron_t fw_gdma_st_out_extend_neuron;
  } fw_tensor_gdma_param_u;

} ir_tensor_gdma_info_t;

} // namespace tpu
} // namespace tpu_mlir

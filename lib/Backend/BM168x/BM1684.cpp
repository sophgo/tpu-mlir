//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

uint32_t BM1684::get_bdc_len(int bdc_num, int group_id) {
  return bdc_num * BDC_CMD_ALIGNED_NUM * sizeof(uint32_t);
}

uint32_t BM1684::get_gdma_len(int gdma_num, int group_id) {
  return gdma_num * GDMA_CMD_ALIGNED_NUM * sizeof(uint32_t);
}

std::shared_ptr<std::vector<int8_t>>
BM1684::Convert1NTo4N(Value v, std::shared_ptr<std::vector<int8_t>> src) {
  auto shape = module::getShape(v);
  const int64_t num_elements = module::getNumElements(v);
  const int32_t N = shape[0];
  const int32_t others = num_elements / N;
  std::shared_ptr<std::vector<int8_t>> dst =
      std::make_shared<std::vector<int8_t>>(align_up(N, 4) * others, 0);
  for (int32_t n = 0; n < N; n++) {
    for (int32_t inner = 0; inner < others; inner++) {
      const int32_t out_idx =
          n / 4 * others * sizeof(int32_t) + (n % 4) + inner * sizeof(int32_t);
      const int32_t in_idx = n * others + inner;
      dst->at(out_idx) = src->at(in_idx);
    }
  }
  return dst;
}

std::shared_ptr<std::vector<int16_t>>
BM1684::Convert1NTo2N(Value v, std::shared_ptr<std::vector<int16_t>> src) {
  auto shape = module::getShape(v);
  const int64_t num_elements = module::getNumElements(v);
  const int32_t N = shape[0];
  const int32_t others = num_elements / N;
  std::shared_ptr<std::vector<int16_t>> dst =
      std::make_shared<std::vector<int16_t>>(align_up(N, 2) * others, 0);
  for (int32_t n = 0; n < N; n++) {
    for (int32_t inner = 0; inner < others; inner++) {
      const int32_t out_idx =
          n / 2 * others * sizeof(int16_t) + (n % 2) + inner * sizeof(int16_t);
      const int32_t in_idx = n * others + inner;
      dst->at(out_idx) = src->at(in_idx);
    }
  }
  return dst;
}

bool BM1684::isL2Load(Operation *op) {
  bool res = false;
  if (auto load_op = dyn_cast_or_null<tpu::LoadOp>(op)) {
    if (module::isWeight(load_op.getInput()) &&
        llvm::any_of(load_op.getOutput().getUsers(), [](Operation *use_op) {
          return isa<tpu::LutOp>(use_op);
        })) {
      res = true;
    }
  }
  return res;
}

void BM1684::load_functions() {
  BM168x::load_functions();
  // clang-format off
  CAST_FUNCTION_WITH_SYM(cmd_id_divide, __cmd_id_divide);
  CAST_FUNCTION_WITH_SYM(cmd_id_merge, __cmd_id_merge);
  CAST_FUNCTION_WITH_SYM(sg_set_profile_dump, bm_set_profile_dump);
  CAST_FUNCTION_WITH_SYM(sg_set_profile_path, bm_set_profile_path);
  CAST_FUNCTION_WITH_SYM(sg_stas_dump, bm_stas_dump);
  CAST_FUNCTION_WITH_SYM(sg_flops_dump, bm_flops_dump);
  CAST_FUNCTION_WITH_SYM(sg_stas_reset, bm_stas_reset);
  CAST_FUNCTION_WITH_SYM(nodechip_depthwise_forward_parallel, nodechip_depthwise_forward_parallel_with_dilation);
  CAST_FUNCTION(tensor_align_move_gen_cmd);
  CAST_FUNCTION(general_matrix_move_gen_cmd);
  CAST_FUNCTION(tensor_general_move_gen_cmd);
  CAST_FUNCTION(nodechip_conv_forward_local);
  CAST_FUNCTION(nodechip_winograd_forward_local);
  CAST_FUNCTION(nodechip_pooling_fix8b_forward_local);
  CAST_FUNCTION(nodechip_conv_forward_local_fix8b);
  CAST_FUNCTION(nodechip_conv_forward_local_fix16b);
  CAST_FUNCTION(nodechip_reduce_full_v3);
  CAST_FUNCTION(nodechip_reduce_full_fix8b);
  CAST_FUNCTION(nodechip_reduce_get_buffer_size_fix8b);
  CAST_FUNCTION(nodechip_winograd_forward_local_fix8b);
  CAST_FUNCTION(nodechip_winograd_double_buffer_forward_local_fix8b);
  CAST_FUNCTION(nodechip_deconv_forward_local);
  CAST_FUNCTION(nodechip_deconv_fix16b_forward_local);
  CAST_FUNCTION(nodechip_deconv_fix8b_forward_local);
  CAST_FUNCTION(nodechip_pooling_forward_local);
  CAST_FUNCTION(nodechip_upsample_forward_local);
  CAST_FUNCTION(nodechip_upsample_fix8b_forward_local);
  CAST_FUNCTION(nodechip_lrn_forward_local);
  CAST_FUNCTION(nodechip_lrn_fix8b_forward_local);
  CAST_FUNCTION(nodechip_batchnorm_layer_local);
  CAST_FUNCTION(nodechip_bnscale_fix8b_forward_local);
  CAST_FUNCTION(nodechip_scale_forward_local);
  CAST_FUNCTION(nodechip_eltwise_forward_local);
  CAST_FUNCTION(nodechip_broadcast_binary_fix8b_forward_parallel);
  CAST_FUNCTION(nodechip_broadcast_binary);
  CAST_FUNCTION(nodechip_broadcast_binary_local);
  CAST_FUNCTION(nodechip_broadcast_binary_fix8b_forward_local);
  CAST_FUNCTION(nodechip_broadcast_binary_full);
  CAST_FUNCTION(nodechip_eltwise_binary_local);
  CAST_FUNCTION(nodechip_eltwise_fix8b_forward_local);
  CAST_FUNCTION(nodechip_fc_forward_local);
  CAST_FUNCTION(nodechip_prelu_forward_local_v2);
  CAST_FUNCTION(nodechip_relu_forward_local);
  CAST_FUNCTION(nodechip_prelu_forward_local_fix8b_v3);
  CAST_FUNCTION(nodechip_relu_forward_local_fix16b);
  CAST_FUNCTION(nodechip_reorg_forward_local);
  CAST_FUNCTION(nodechip_reorg_forward_fix8b_local);
  CAST_FUNCTION(nodechip_permute_forward_local);
  CAST_FUNCTION(nodechip_permute_fix8b_forward_local);
  CAST_FUNCTION(nodechip_normalize_forward_local);
  CAST_FUNCTION(nodechip_normalize_fix8b_forward_local);
  CAST_FUNCTION(nodechip_active_forward_local);
  CAST_FUNCTION(nodechip_mulshift_fix8b_forward_local);
  CAST_FUNCTION(nodechip_concat_md);
  CAST_FUNCTION(nodechip_concat_md_fix8b);
  CAST_FUNCTION(nodechip_stride_slice_forward_local);
  CAST_FUNCTION(nodechip_stride_slice_forward_local_fix8b);
  CAST_FUNCTION(nodechip_pooling3d_local);
  CAST_FUNCTION(nodechip_conv3d_local);
  CAST_FUNCTION(nodechip_conv3d_fix8b_local);
  CAST_FUNCTION(nodechip_deconv3d_local);
  CAST_FUNCTION(nodechip_fc_forward_parallel);
  CAST_FUNCTION(nodechip_fc_fix8b_forward_parallel);
  CAST_FUNCTION(nodechip_pooling_forward_parallel_with_data_split);
  CAST_FUNCTION(nodechip_pooling_fix8b_forward_parallel_with_data_split);
  CAST_FUNCTION(nodechip_pooling_tf_fix8b_forward_parallel_with_data_split);
  CAST_FUNCTION(nodechip_sort_per_dim);
  CAST_FUNCTION(nodechip_sort_per_dim_fix8b);
  CAST_FUNCTION(nodechip_index_select);
  CAST_FUNCTION(nodechip_index_select_fix8b);
  CAST_FUNCTION(nodechip_psroipooling_forward_with_datasplit);
  CAST_FUNCTION(nodechip_psroipooling_fix8b_forward_with_datasplit);
  CAST_FUNCTION(nodechip_roi_pooling_forward);
  CAST_FUNCTION(nodechip_crop);
  CAST_FUNCTION(nodechip_crop_fix8b);
  CAST_FUNCTION(nodechip_upsample_forward_parallel_with_data_split);
  CAST_FUNCTION(nodechip_upsample_forward_parallel_fix8b);
  CAST_FUNCTION(nodechip_upsample_mask_forward);
  CAST_FUNCTION(nodechip_multiregion_forward_parallel);
  CAST_FUNCTION(nodechip_deconv_forward_parallel_with_data_split);
  CAST_FUNCTION(nodechip_deconv_forward_parallel_with_data_split_v2);
  CAST_FUNCTION(nodechip_deconv_fix16b_forward_parallel);
  CAST_FUNCTION(nodechip_deconv_fix8b_forward_parallel);
  CAST_FUNCTION(nodechip_conv_forward_parallel_with_data_split);
  CAST_FUNCTION(nodechip_winograd_forward_parallel_with_data_split);
  CAST_FUNCTION(nodechip_winograd_forward_parallel_fix8b_with_data_split);
  CAST_FUNCTION(nodechip_lstm_forward_parallel);
  CAST_FUNCTION(nodechip_scale_forward);
  CAST_FUNCTION(nodechip_eltwise_forward);
  CAST_FUNCTION(nodechip_eltwise_fix8b_forward_parallel);
  CAST_FUNCTION(nodechip_prelu_forward);
  CAST_FUNCTION(nodechip_prelu_forward_fix8b);
  CAST_FUNCTION(nodechip_relu_forward_fix16b);
  CAST_FUNCTION(nodechip_permute_forward);
  CAST_FUNCTION(nodechip_permute_fix8b_forward);
  CAST_FUNCTION(nodechip_normalize_forward);
  CAST_FUNCTION(nodechip_normalize_fix8b_forward);
  CAST_FUNCTION(nodechip_slice_forward);
  CAST_FUNCTION(nodechip_softmax_forward_parallel);
  CAST_FUNCTION(nodechip_active_forward_parallel);
  CAST_FUNCTION(nodechip_active_forward_parallel_fix8b);
  CAST_FUNCTION(nodechip_relu_forward_32bit_parallel);
  CAST_FUNCTION(nodechip_batchnorm_forward_inference_parallel);
  CAST_FUNCTION(nodechip_batchnorm_forward_inference_parallel_v2);
  CAST_FUNCTION(nodechip_bnscale_forward_parallel_fix8b_with_src_storage_mode);
  CAST_FUNCTION(nodechip_lrn_forward_parallel);
  CAST_FUNCTION(nodechip_lrn_fix8b_forward_parallel);
  CAST_FUNCTION(nodechip_depthwise_fix8b_forward_parallel);
  CAST_FUNCTION(nodechip_conv_forward_parallel_fix8b_with_data_split);
  CAST_FUNCTION(nodechip_conv_forward_parallel_fix16b_with_data_split);
  CAST_FUNCTION(nodechip_mulshift_fix8b_forward);
  CAST_FUNCTION(nodechip_global_conv_data_split_fix8b);
  CAST_FUNCTION(nodechip_rpnproposal_forward);
  CAST_FUNCTION(nodechip_shuffle_channel_forward);
  CAST_FUNCTION(nodechip_shuffle_channel_fix8b_forward);
  CAST_FUNCTION(nodechip_topk);
  CAST_FUNCTION(nodechip_topk_fix8b);
  CAST_FUNCTION(nodechip_lut_v2);
  CAST_FUNCTION(nodechip_cumsum);
  CAST_FUNCTION(nodechip_arg);
  CAST_FUNCTION(nodechip_arg_local);
  CAST_FUNCTION(nodechip_arg_fix8b);
  CAST_FUNCTION(nodechip_stride_slice_md);
  CAST_FUNCTION(nodechip_stride_slice_fix8b);
  CAST_FUNCTION(nodechip_split_tf_md);
  CAST_FUNCTION(nodechip_split_tf_fix8b_md);
  CAST_FUNCTION(nodechip_swap_dim);
  CAST_FUNCTION(nodechip_swap_dim_fix8b);
  CAST_FUNCTION(nodechip_interp_forward_parallel);
  CAST_FUNCTION(nodechip_interp_forward_fix8b_parallel);
  CAST_FUNCTION(nodechip_reverse_forward_v2);
  CAST_FUNCTION(nodechip_reorg_forward_v2);
  CAST_FUNCTION(nodechip_reorg_forward_fix8b);
  CAST_FUNCTION(nodechip_adaptive_pooling_forward);
  CAST_FUNCTION(nodechip_yolo);
  CAST_FUNCTION(nodechip_memset);
  CAST_FUNCTION(nodechip_batch_matmul_forward_parallel_v2);
  CAST_FUNCTION(nodechip_channel_shift_forward);
  CAST_FUNCTION(nodechip_channel_shift_forward_fix8b);
  CAST_FUNCTION(nodechip_interleave);
  CAST_FUNCTION(nodechip_interleave_fix8b);
  CAST_FUNCTION(nodechip_interleave_local);
  CAST_FUNCTION(nodechip_interleave_fixpoint_local);
  CAST_FUNCTION(nodechip_stride_calculate_forward);
  CAST_FUNCTION(nodechip_stridecalc_forward_global);
  CAST_FUNCTION(nodechip_stride_calculate_forward_fix8b);
  CAST_FUNCTION(nodechip_stridecalc_forward_local);
  CAST_FUNCTION(nodechip_stridecalc_fix8b_forward_global);
  CAST_FUNCTION(nodechip_stridecalc_fix8b_forward_local);
  CAST_FUNCTION(nodechip_bitwise_forward_global);
  CAST_FUNCTION(nodechip_bitwise_forward_local);
  CAST_FUNCTION(nodechip_arith_shift_forward);
  CAST_FUNCTION(nodechip_arith_shift_forward_local_v2);
  CAST_FUNCTION(nodechip_reshape_fix8b);
  CAST_FUNCTION(nodechip_pooling3d_forward_parallel);
  CAST_FUNCTION(nodechip_pooling3d_fix8b_forward_parallel);
  CAST_FUNCTION(nodechip_max_pooling_with_mask_forward_v2);
  CAST_FUNCTION(nodechip_conv3d_parallel);
  CAST_FUNCTION(nodechip_deconv3d);
  CAST_FUNCTION(nodechip_conv3d_fix8b_parallel);
  CAST_FUNCTION(nodechip_conv3d_add_parallel);
  CAST_FUNCTION(nodechip_unfold);
  CAST_FUNCTION(nodechip_gru);
  CAST_FUNCTION(nodechip_pytorch_lstm);
  CAST_FUNCTION(nodechip_matrix_band_part);
  CAST_FUNCTION(nodechip_global_memcpy_ex);
  CAST_FUNCTION(nodechip_lut_local_v2);
  CAST_FUNCTION(nodechip_serial_number_gen);
  CAST_FUNCTION(nodechip_pad_fix8b);
  CAST_FUNCTION(nodechip_pad);
  CAST_FUNCTION(nodechip_pad_local);
  CAST_FUNCTION(nodechip_pad_fix8b_local);
  CAST_FUNCTION(nodechip_pad3d);
  CAST_FUNCTION(nodechip_pad3d_fix8b);
  CAST_FUNCTION(nodechip_concat_local_v2);
  CAST_FUNCTION(nodechip_concat_fix8b_local_v2);
  CAST_FUNCTION(nodechip_const_binary);
  CAST_FUNCTION(nodechip_global_int2float);
  CAST_FUNCTION(nodechip_float2int8_v2);
  CAST_FUNCTION(nodechip_const_binary_local);
  CAST_FUNCTION(nodechip_const_binary_fix8b_forward_local);
  CAST_FUNCTION(nodechip_const_binary_fix8b_forward_parallel);
  CAST_FUNCTION(nodechip_transpose);
  CAST_FUNCTION(nodechip_transpose_fix8b);
  CAST_FUNCTION(nodechip_float2int8_local_keep_input);
  CAST_FUNCTION(tensor_int8_to_float_local_v2);
  CAST_FUNCTION(get_broadcast_binary_buffer_size);
  CAST_FUNCTION(nodechip_group_norm);
  CAST_FUNCTION(nodechip_group_norm_local);
  CAST_FUNCTION(nodechip_depth2space_mlir);
  CAST_FUNCTION(nodechip_tile_full);
  CAST_FUNCTION(nodechip_tile_full_fix8b);
  CAST_FUNCTION(nodechip_tile_local);
  CAST_FUNCTION(nodechip_tile_fix8b_local);
  CAST_FUNCTION(nodechip_space2batch);
  CAST_FUNCTION(nodechip_space2batch_fix8b);
  CAST_FUNCTION(nodechip_batch2space);
  CAST_FUNCTION(nodechip_batch2space_fix8b);
  CAST_FUNCTION(nodechip_unary);
  CAST_FUNCTION(nodechip_masked_fill_global);
  CAST_FUNCTION(nodechip_masked_fill_local);
  CAST_FUNCTION(nodechip_float_to_int32_global);
  CAST_FUNCTION(nodechip_float_to_int32_local);
  CAST_FUNCTION(nodechip_deconv_forward_local_v2);
  CAST_FUNCTION(nodechip_select_all);
  CAST_FUNCTION(nodechip_select_fix8b);
  CAST_FUNCTION(nodechip_select_local);
  CAST_FUNCTION(nodechip_select_fix8b_local);
  CAST_FUNCTION(allow_store_cmd);
  // clang-format on
}

unsigned int BM1684::get_total_id(const char *engine_name) {
  if (strcmp(engine_name, "tiu:0:0") == 0)
    return bdc_total_id;
  else if (strcmp(engine_name, "gdma:0:0") == 0)
    return gdma_total_id;
  return 0;
}

unsigned int BM1684::get_inst_number_per_group(const char *engine_name,
                                               int group_idx) {
  if (strcmp(engine_name, "tiu:0:0") == 0)
    return bdc_group_id[group_idx];
  else if (strcmp(engine_name, "gdma:0:0") == 0)
    return gdma_group_id[group_idx];
  return 0;
}

unsigned int BM1684::get_group_number() { return cmdid_groupnum; }
const unsigned char *BM1684::get_inst_data(const char *engine_name) {
  if (strcmp(engine_name, "tiu:0:0") == 0)
    return (const unsigned char *)bdc_buffer.data();
  else if (strcmp(engine_name, "gdma:0:0") == 0)
    return (const unsigned char *)gdma_buffer.data();
  return nullptr;
}

void BM1684::before_codegen() {
  BM168x::before_codegen();
  gdma_group_id.clear();
  gdma_group_id.push_back(0);
  bdc_total_id = 0;
  gdma_total_id = 0;
  bdc_group_id.clear();
  bdc_group_id.push_back(0);
  gdma_bytes.clear();
  bdc_bytes.clear();
  gdma_buffer.clear();
  bdc_buffer.clear();
  cmdid_groupnum = 1;
}

void BM1684::start_env() {
  BM168x::start_env();
  gdma_buffer.reserve(0x1000000);
  bdc_buffer.reserve(0x1000000);
  dl_set_cmd_buffer_ptr((void *)&gdma_buffer, (void *)&bdc_buffer);
  dl_set_total_id_ptr(&gdma_total_id, &bdc_total_id, code->cmdid_node,
                      (void *)&gdma_group_id, (void *)&bdc_group_id,
                      &cmdid_groupnum);
  dl_allow_store_cmd();
  dl_forbid_atomic_cmodel();
}

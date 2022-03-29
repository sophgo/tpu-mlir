#include "sophgo/Backend/BM1684.h"
#include "sophgo/Support/Helper/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

using namespace sophgo::backend;
using namespace sophgo::helper;
using namespace mlir;

template <typename FPtrTy> FPtrTy BM1684::CastToFPtr(const char *symbolName) {
  assert(DL.isValid());
  auto fPtr = DL.getAddressOfSymbol(symbolName);
  if (fPtr == nullptr) {
    llvm::errs() << "can't find symbol: " << symbolName << "\n";
    llvm_unreachable(symbolName);
  }
  return reinterpret_cast<FPtrTy>(fPtr);
}

void *BM1684::get_gmem_addr(uint64_t addr) {
  auto start = static_cast<char *>(this->dl_get_global_memaddr(0));
  return start + addr - GLOBAL_MEM_START;
}

void *BM1684::get_gmem_addr(const bm_device_mem_t &mem) {
  auto start = static_cast<char *>(this->dl_get_global_memaddr(0));
  return start + mem.offset;
}

void BM1684::bm_memcpy_s2d(const bm_device_mem_t &dst, void *src) {
  memcpy(get_gmem_addr(dst), src, dst.size);
}

void BM1684::bm_memcpy_d2s(void *dst, const bm_device_mem_t &src) {
  memcpy(dst, get_gmem_addr(src), src.size);
}

void BM1684::value_s2d(Value v, void *src) {
  auto addr = Module::getAddress(v);
  auto bytes = Module::getBytes(v);
  memcpy(get_gmem_addr(addr), src, bytes);
}
void BM1684::value_d2s(Value v, void *dst) {
  auto addr = Module::getAddress(v);
  auto bytes = Module::getBytes(v);
  memcpy(dst, get_gmem_addr(addr), bytes);
}

void BM1684::init() {
  // setup
  dl_cmodel_init(0, CMODEL_GLOBAL_MEM_SIZE);
  cmdid_node = dl_create_cmd_id_node();
  dl_reset_cmd_id(cmdid_node);
  set_command_issue_flag(true);
  bdc_buffer = std::make_shared<std::vector<uint32_t>>(0x1000000);
  gdma_buffer = std::make_shared<std::vector<uint32_t>>(0x1000000);
  gdma_group_id.clear();
  gdma_group_id.push_back(0);
  bdc_group_id.clear();
  bdc_group_id.push_back(0);
  cmdid_groupnum = 1;
  dl_set_cmd_buffer_ptr((void *)gdma_buffer->data(),
                        (void *)bdc_buffer->data());
  dl_set_total_id_ptr(&gdma_total_id, &bdc_total_id, cmdid_node,
                      (void *)&gdma_group_id, (void *)&bdc_group_id,
                      &cmdid_groupnum);
}

void BM1684::deinit() {
  if (DL.isValid()) {
    if (cmdid_node != nullptr) {
      dl_destroy_cmd_id_node(cmdid_node);
    }
    dl_cmodel_deinit(0);
  }
}

bm_data_type_t BM1684::getType(mlir::Type type) {
  if (type.isF32()) {
    return DTYPE_FP32;
  }
  if (type.isBF16()) {
    return DTYPE_BFP16;
  }
  if (type.isF16()) {
    return DTYPE_FP16;
  }
  if (type.isSignedInteger(8)) {
    return DTYPE_INT8;
  }
  if (type.isSignedInteger(16)) {
    return DTYPE_INT16;
  }
  if (type.isSignedInteger(32)) {
    return DTYPE_INT32;
  }
  if (type.isUnsignedInteger(8)) {
    return DTYPE_UINT8;
  }
  if (type.isUnsignedInteger(16)) {
    return DTYPE_UINT16;
  }
  if (type.isUnsignedInteger(32)) {
    return DTYPE_UINT32;
  }
  type.dump();
  llvm_unreachable("unknow type");
  return DTYPE_FP32;
}

#define CAST_FUNCTION(name) dl_##name = CastToFPtr<name>(#name)

BM1684::BM1684() {
  std::string Err;
  DL = llvm::sys::DynamicLibrary::getPermanentLibrary("libbackend_1684.so",
                                                      &Err);
  if (DL.isValid() == false) {
    llvm_unreachable(Err.c_str());
  }
  CAST_FUNCTION(cmodel_init);
  CAST_FUNCTION(cmodel_deinit);
  CAST_FUNCTION(create_cmd_id_node);
  CAST_FUNCTION(destroy_cmd_id_node);
  CAST_FUNCTION(set_cmd_id_cycle);
  CAST_FUNCTION(get_cmd_id_cycle);
  CAST_FUNCTION(reset_cmd_id);
  CAST_FUNCTION(allow_store_cmd);
  CAST_FUNCTION(forbid_store_cmd);
  CAST_FUNCTION(use_atomic_cmodel);
  CAST_FUNCTION(forbid_atomic_cmodel);
  CAST_FUNCTION(get_global_memaddr);
  CAST_FUNCTION(set_cmd_buffer_ptr);
  CAST_FUNCTION(set_total_id_ptr);
  CAST_FUNCTION(tensor_align_move_gen_cmd);
  CAST_FUNCTION(general_matrix_move_gen_cmd);
  CAST_FUNCTION(nodechip_conv_forward_local);
  CAST_FUNCTION(nodechip_winograd_forward_local);
  CAST_FUNCTION(nodechip_pooling_fix8b_forward_local);
  CAST_FUNCTION(nodechip_conv_forward_local_fix8b);
  CAST_FUNCTION(nodechip_conv_forward_local_fix16b);
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
  CAST_FUNCTION(nodechip_deconv_fix16b_forward_parallel);
  CAST_FUNCTION(nodechip_deconv_fix8b_forward_parallel);
  CAST_FUNCTION(nodechip_depthwise_forward_parallel_with_dilation);
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
  CAST_FUNCTION(nodechip_interp_forward_parallel);
  CAST_FUNCTION(nodechip_interp_forward_fix8b_parallel);
  CAST_FUNCTION(nodechip_reverse_forward_v2);
  CAST_FUNCTION(nodechip_reorg_forward_v2);
  CAST_FUNCTION(nodechip_reorg_forward_fix8b);
  CAST_FUNCTION(nodechip_adaptive_pooling_forward);
  CAST_FUNCTION(nodechip_yolo);
  CAST_FUNCTION(nodechip_memset);
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
  CAST_FUNCTION(nodechip_concat_local_v2);
  CAST_FUNCTION(nodechip_concat_fix8b_local_v2);
  CAST_FUNCTION(nodechip_const_binary);
  CAST_FUNCTION(nodechip_global_int2float);
  CAST_FUNCTION(nodechip_float2int8_v2);
}

BM1684::~BM1684() {
}

void BM1684::reset_cmd_id_node() { dl_reset_cmd_id(cmdid_node); }

void BM1684::set_command_issue_flag(bool value) {
  really_issue_command = value;
  if (really_issue_command) {
    dl_allow_store_cmd();
    dl_use_atomic_cmodel();
  } else {
    dl_forbid_store_cmd();
    dl_forbid_atomic_cmodel();
  }
}

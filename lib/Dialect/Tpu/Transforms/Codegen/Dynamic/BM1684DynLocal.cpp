//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/BM1684DynLocal.hpp"
#define IR_PACKING
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/BM1684DynIrUtils.hpp"

using namespace llvm;

using namespace tpu_mlir::backend;
using namespace std;

namespace tpu_mlir {
namespace tpu {

void *call_local_layer_ir_write(FW_LAYER_TYPE_T fw_layer_type, void *p_ir_addr,
                                ir_layer_info_t *p_ir_layer_info) {
  uint64_t pre_ir_addr = 0;
  if (fw_layer_type == FW_BMNET_CONV) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_conv_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_conv_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_POOL) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_pool_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_pooling_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_LRN) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_lrn_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_lrn_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_PRELU) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_prelu_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_prelu_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_ELTWISE) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_eltwise_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_eltwise_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_SCALE) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_scale_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_scale_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_BATCHNORM) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_batchnorm_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_batchnorm_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_DECONV) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_deconv_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_deconv_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_MULSHIFT) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_mulshift_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_mulshift_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_CONCAT) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_concat_loc_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_concat_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_POOLTF) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_pooltf_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_pooling_tf_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_ACTIVE) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_active_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_active_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_CROP) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_crop_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_crop_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_BROADCAST_BINARY) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_broadcast_binary_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr =
        static_loc_broadcast_binary_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_ELTWISE_BINARY) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_eltwise_binary_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr =
        static_loc_eltwise_binary_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_CONST_BINARY) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_const_binary_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_const_binary_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_BIASADD) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_biasadd_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_biasadd_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_SELECT) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_select_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_select_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_NORMALIZE) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_normalize_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_normalize_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_UPSAMPLE) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_upsample_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_upsample_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_STRIDESLICE) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_stride_slice_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_strideslice_irbuf_write(p_ir_addr, p_ir_layer_info);
    // local_dynamic step 6: add fw_layer_type case for irbuf write
  } else if (fw_layer_type == FW_BMNET_ARITH_SHIFT) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_arith_shift_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_arith_shift_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_REORG) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_reorg_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_reorg_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_DTYPE_CONVERT) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_dtype_convert_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr =
        static_loc_dtype_convert_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_TILE) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_tile_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_tile_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_PAD) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_pad_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_pad_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_REDUCE_FULL) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_reduce_full_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_reduce_full_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_LUT) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_lut_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_lut_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_STRIDECALC) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_stridecalc_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_stridecalc_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_INTERLEAVE) {
    *(u32 *)p_ir_addr = (u32)sizeof(fw_interleave_layer_param_t);
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_interleave_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_BITWISE) {
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_bitwise_layer_param_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_bitwise_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_BINARY_SHIFT) {
    *(u32 *)p_ir_addr = (u32)sizeof(fw_binary_shift_layer_param_t);
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_binary_shift_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_ARG) {
    *(u32 *)p_ir_addr = (u32)sizeof(fw_arg_layer_param_t);
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_arg_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_TPU) {
    *(u32 *)p_ir_addr = (u32)sizeof(fw_arg_layer_param_t);
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_tpu_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_CONV3D) {
    *(u32 *)p_ir_addr = (u32)sizeof(fw_conv3d_layer_param_t);
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_conv3d_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_POOL3D) {
    *(u32 *)p_ir_addr = (u32)sizeof(fw_pool3d_layer_param_t);
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_pool3d_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else if (fw_layer_type == FW_BMNET_GROUP_NORM) {
    *(u32 *)p_ir_addr = (u32)sizeof(fw_group_norm_layer_param_t);
    p_ir_addr = (u32 *)p_ir_addr + 1;
    pre_ir_addr = (u64)p_ir_addr;
    p_ir_addr = static_loc_group_norm_irbuf_write(p_ir_addr, p_ir_layer_info);
  } else {
    std::cout << "not supported fw local layer type " << fw_layer_type
              << std::endl;
    llvm_unreachable("fw_layer_type error");
  }
  if ((u32)((u64)p_ir_addr - pre_ir_addr) !=
      p_ir_layer_info->layer_fw_ir_length) {
    llvm::errs()
        << "write ir buff size is not same with fw_ir_length, fw local "
           "layer type is "
        << fw_layer_type;
    llvm_unreachable("ir length error");
  }
  return p_ir_addr;
}

void *static_loc_mulshift_irbuf_write(void *p_ir_buf,
                                      ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  u32 bottom_tensor_id;
  u32 top_tensor_id;

  fw_mulshift_layer_param_t fw_mulshift_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_mulshift_layer_param;

  *(fw_mulshift_layer_param_t *)p_ir_addr = fw_mulshift_layer_param;
  p_ir_addr = (fw_mulshift_layer_param_t *)p_ir_addr + 1;

  bottom_tensor_id = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  top_tensor_id = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;
  *(u32 *)p_ir_addr = (bottom_tensor_id << 16) | (top_tensor_id & 0xffff);
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  int idx = 2;
  // imm buffer for specific depthwise conv
  if (p_ir_layer_info->ir_tensor_info_v.size() > (u32)idx) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[idx].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  // output consumer number info
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].consumer_number;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_conv_irbuf_write(void *p_ir_buf,
                                  ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  u32 bottom_tensor_id;
  u32 top_tensor_id = 0;
  u32 scale_tensor_id = 0;

  fw_conv_layer_param_t fw_conv_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_conv_layer_param;

  *(fw_conv_layer_param_t *)p_ir_addr = fw_conv_layer_param;
  p_ir_addr = (fw_conv_layer_param_t *)p_ir_addr + 1;

  bottom_tensor_id = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;

  u32 *top_bottom_id_p = (u32 *)p_ir_addr;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  int idx = 2;
  if (fw_conv_layer_param.using_bias == 1) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[idx].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    idx++;
  }

  if (fw_conv_layer_param.if_batchnorm == 1) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[idx].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    idx++;
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[idx].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    idx++;
  }

  if (fw_conv_layer_param.if_scale == 1) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[idx].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    scale_tensor_id = p_ir_layer_info->ir_tensor_info_v[idx].tensor_id;
    idx++;
    if (fw_conv_layer_param.scale_bias == 1) {
      *(u32 *)p_ir_addr =
          p_ir_layer_info->ir_tensor_info_v[idx].local_mem_offset;
      p_ir_addr = (u32 *)p_ir_addr + 1;
      idx++;
    }
  }

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[idx].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  top_tensor_id = p_ir_layer_info->ir_tensor_info_v[idx].tensor_id;
  u32 out_consumer_num = p_ir_layer_info->ir_tensor_info_v[idx].consumer_number;

  idx++;
  *(u32 *)top_bottom_id_p = (bottom_tensor_id << 16) | (top_tensor_id & 0xffff);

  if (fw_conv_layer_param.if_scale == 1) {
    *(u32 *)p_ir_addr = scale_tensor_id;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  // imm buffer for specific depthwise conv
  if (p_ir_layer_info->ir_tensor_info_v.size() > (u32)idx) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[idx].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  // output consumer number info
  *(u32 *)p_ir_addr = out_consumer_num;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_pooling_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  u32 bottom_tensor_id;
  u32 top_tensor_id;

  fw_pool_layer_param_t fw_pool_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_pool_layer_param;

  *(fw_pool_layer_param_t *)p_ir_addr = fw_pool_layer_param;
  p_ir_addr = (fw_pool_layer_param_t *)p_ir_addr + 1;

  bottom_tensor_id = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  top_tensor_id = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;

  *(u32 *)p_ir_addr = (bottom_tensor_id << 16) | (top_tensor_id & 0xffff);
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // output consumer number info
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].consumer_number;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_pooling_tf_irbuf_write(void *p_ir_buf,
                                        ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  u32 bottom_tensor_id;
  u32 top_tensor_id;

  fw_pooltf_layer_param_t fw_pooltf_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_pooltf_layer_param;

  *(fw_pooltf_layer_param_t *)p_ir_addr = fw_pooltf_layer_param;
  p_ir_addr = (fw_pooltf_layer_param_t *)p_ir_addr + 1;

  bottom_tensor_id = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  top_tensor_id = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;

  *(u32 *)p_ir_addr = (bottom_tensor_id << 16) | (top_tensor_id & 0xffff);
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // output consumer number info
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].consumer_number;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_lrn_irbuf_write(void *p_ir_buf,
                                 ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  u32 bottom_tensor_id;
  u32 top_tensor_id;

  fw_lrn_layer_param_t fw_lrn_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_lrn_layer_param;

  *(fw_lrn_layer_param_t *)p_ir_addr = fw_lrn_layer_param;
  p_ir_addr = (fw_lrn_layer_param_t *)p_ir_addr + 1;

  bottom_tensor_id = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  top_tensor_id = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;

  *(u32 *)p_ir_addr = (bottom_tensor_id << 16) | (top_tensor_id & 0xffff);
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // output consumer number info
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].consumer_number;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_prelu_irbuf_write(void *p_ir_buf,
                                   ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  u32 bottom_tensor_id;
  u32 top_tensor_id;

  fw_prelu_layer_param_t fw_prelu_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_prelu_layer_param;

  *(fw_prelu_layer_param_t *)p_ir_addr = fw_prelu_layer_param;
  p_ir_addr = (fw_prelu_layer_param_t *)p_ir_addr + 1;

  bottom_tensor_id = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  if (fw_prelu_layer_param.channel_shared) {
    top_tensor_id = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;
  } else {
    top_tensor_id = p_ir_layer_info->ir_tensor_info_v[2].tensor_id;
  }
  *(u32 *)p_ir_addr = (bottom_tensor_id << 16) | (top_tensor_id & 0xffff);
  p_ir_addr = (u32 *)p_ir_addr + 1;

  int idx = 0;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[idx++].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  if (fw_prelu_layer_param.channel_shared == 0) {
    *(u32 *)p_ir_addr =
        p_ir_layer_info->ir_tensor_info_v[idx++].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[idx].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  u32 out_consumer_num = p_ir_layer_info->ir_tensor_info_v[idx].consumer_number;
  idx++;

  if (!(fw_prelu_layer_param.channel_shared &&
        fw_prelu_layer_param.shared_slope == 0)) {
    *(u32 *)p_ir_addr =
        p_ir_layer_info->ir_tensor_info_v[idx++].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  // output consumer number info
  *(u32 *)p_ir_addr = out_consumer_num;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_eltwise_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_eltwise_layer_param_t fw_eltwise_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_eltwise_layer_param;

  *(fw_eltwise_layer_param_t *)p_ir_addr = fw_eltwise_layer_param;
  p_ir_addr = (fw_eltwise_layer_param_t *)p_ir_addr + 1;
  // input
  for (u8 i = 0; i < fw_eltwise_layer_param.in_num; ++i) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[i].tensor_id;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    if (fw_eltwise_layer_param.op_code == 1) {
      *(float *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[i].coeff;
      p_ir_addr = (float *)p_ir_addr + 1;
    }
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[i].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }
  // output
  *(u32 *)p_ir_addr =
      p_ir_layer_info->ir_tensor_info_v[fw_eltwise_layer_param.in_num]
          .tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr =
      p_ir_layer_info->ir_tensor_info_v[fw_eltwise_layer_param.in_num]
          .local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  if (p_ir_layer_info->ir_tensor_info_v.size() >
      (u32)(fw_eltwise_layer_param.in_num + 1)) {
    *(u32 *)p_ir_addr =
        p_ir_layer_info->ir_tensor_info_v[fw_eltwise_layer_param.in_num + 1]
            .local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  // output consumer number info
  *(u32 *)p_ir_addr =
      p_ir_layer_info->ir_tensor_info_v[fw_eltwise_layer_param.in_num]
          .consumer_number;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_scale_irbuf_write(void *p_ir_buf,
                                   ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  u32 bottom_tensor_id;
  u32 top_tensor_id;
  u32 top_consumer_num = 1;

  fw_scale_layer_param_t fw_scale_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_scale_layer_param;

  *(fw_scale_layer_param_t *)p_ir_addr = fw_scale_layer_param;
  p_ir_addr = (fw_scale_layer_param_t *)p_ir_addr + 1;

  bottom_tensor_id = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  if (fw_scale_layer_param.using_bias) {
    top_tensor_id = p_ir_layer_info->ir_tensor_info_v[3].tensor_id;
    top_consumer_num = p_ir_layer_info->ir_tensor_info_v[3].consumer_number;
  } else {
    top_tensor_id = p_ir_layer_info->ir_tensor_info_v[2].tensor_id;
    top_consumer_num = p_ir_layer_info->ir_tensor_info_v[2].consumer_number;
  }

  *(u32 *)p_ir_addr = (bottom_tensor_id << 16) | (top_tensor_id & 0xffff);
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  if (fw_scale_layer_param.using_bias) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[3].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    if (p_ir_layer_info->ir_tensor_info_v.size() > 4) {
      *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[4].local_mem_offset;
      p_ir_addr = (u32 *)p_ir_addr + 1;
      *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[5].local_mem_offset;
      p_ir_addr = (u32 *)p_ir_addr + 1;
    }
  } else {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    if (p_ir_layer_info->ir_tensor_info_v.size() > 3) {
      *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[3].local_mem_offset;
      p_ir_addr = (u32 *)p_ir_addr + 1;
      *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[4].local_mem_offset;
      p_ir_addr = (u32 *)p_ir_addr + 1;
    }
  }
  // output consumer number info
  *(u32 *)p_ir_addr = top_consumer_num;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_batchnorm_irbuf_write(void *p_ir_buf,
                                       ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  u32 bottom_tensor_id;
  u32 top_tensor_id;
  u32 scale_tensor_id;
  u32 top_consumer_num = 1;

  fw_batchnorm_layer_param_t fw_batchnorm_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_batchnorm_layer_param;

  int non_scale_size = p_ir_layer_info->ir_tensor_info_v.size();
  if (fw_batchnorm_layer_param.if_scale && fw_batchnorm_layer_param.scale_bias)
    non_scale_size -= 2;
  else if (fw_batchnorm_layer_param.if_scale &&
           !fw_batchnorm_layer_param.scale_bias)
    non_scale_size -= 1;
  else
    non_scale_size = non_scale_size;

  *(fw_batchnorm_layer_param_t *)p_ir_addr = fw_batchnorm_layer_param;
  p_ir_addr = (fw_batchnorm_layer_param_t *)p_ir_addr + 1;

  bottom_tensor_id = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;

  top_tensor_id = p_ir_layer_info->ir_tensor_info_v[3].tensor_id;
  top_consumer_num = p_ir_layer_info->ir_tensor_info_v[3].consumer_number;

  *(u32 *)p_ir_addr = (bottom_tensor_id << 16) | (top_tensor_id & 0xffff);
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // output
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[3].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  int idx = 4;
  if (fw_batchnorm_layer_param.if_scale == 1) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[idx].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    scale_tensor_id = p_ir_layer_info->ir_tensor_info_v[idx].tensor_id;
    idx++;
    if (fw_batchnorm_layer_param.scale_bias == 1) {
      *(u32 *)p_ir_addr =
          p_ir_layer_info->ir_tensor_info_v[idx].local_mem_offset;
      p_ir_addr = (u32 *)p_ir_addr + 1;
      idx++;
    }
  }

  if (fw_batchnorm_layer_param.if_scale == 1) {
    *(u32 *)p_ir_addr = scale_tensor_id;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  if (non_scale_size > 4) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[idx].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    idx++;
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[idx].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  // output consumer number info
  *(u32 *)p_ir_addr = top_consumer_num;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_deconv_irbuf_write(void *p_ir_buf,
                                    ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  u32 bottom_tensor_id;
  u32 top_tensor_id;
  u32 top_consumer_num = 1;

  fw_deconv_layer_param_t fw_deconv_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_deconv_layer_param;

  *(fw_deconv_layer_param_t *)p_ir_addr = fw_deconv_layer_param;
  p_ir_addr = (fw_deconv_layer_param_t *)p_ir_addr + 1;

  bottom_tensor_id = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  if (fw_deconv_layer_param.using_bias) {
    top_tensor_id = p_ir_layer_info->ir_tensor_info_v[3].tensor_id;
    top_consumer_num = p_ir_layer_info->ir_tensor_info_v[3].consumer_number;
  } else {
    top_tensor_id = p_ir_layer_info->ir_tensor_info_v[2].tensor_id;
    top_consumer_num = p_ir_layer_info->ir_tensor_info_v[2].consumer_number;
  }
  *(u32 *)p_ir_addr = (bottom_tensor_id << 16) | (top_tensor_id & 0xffff);
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  int idx = 2;
  if (fw_deconv_layer_param.using_bias) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[idx].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    idx++;
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[idx].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    idx++;
  } else {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[idx].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    idx++;
  }

  // output consumer number info
  *(u32 *)p_ir_addr = top_consumer_num;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_concat_irbuf_write(void *p_ir_buf,
                                    ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_concat_loc_layer_param_t fw_concat_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_concat_loc_layer_param;

  *(fw_concat_loc_layer_param_t *)p_ir_addr = fw_concat_layer_param;
  p_ir_addr = (fw_concat_loc_layer_param_t *)p_ir_addr + 1;

  u32 bottom_tensors_num = p_ir_layer_info->ir_tensor_info_v.size() - 1;
  vector<u32> bottom_tensors_id;
  vector<u32> bottom_tensors_local_offset;

  u32 top_tensor_id = 0;
  u32 top_tensor_loc_offset = 0;
  u32 top_consumer_num = 1;
  for (auto it = p_ir_layer_info->ir_tensor_info_v.begin();
       it != p_ir_layer_info->ir_tensor_info_v.end(); it++) {
    if (it == p_ir_layer_info->ir_tensor_info_v.begin()) {
      top_tensor_id = it->tensor_id;
      top_tensor_loc_offset = it->local_mem_offset;
      top_consumer_num = it->consumer_number;
    } else {
      bottom_tensors_id.push_back(it->tensor_id);
      bottom_tensors_local_offset.push_back(it->local_mem_offset);
    }
  }

  *(u32 *)p_ir_addr = (top_tensor_id << 16) | (bottom_tensors_num & 0xffff);
  p_ir_addr = (u32 *)p_ir_addr + 1;

  for (u32 i = 0; i < bottom_tensors_id.size(); i = i + 2) {
    u32 in_tensor_A = bottom_tensors_id[i];
    u32 in_tensor_B = 0;
    if (i + 1 < bottom_tensors_id.size())
      in_tensor_B = bottom_tensors_id[i + 1];
    *(u32 *)p_ir_addr = (in_tensor_B << 16) | (in_tensor_A & 0xffff);
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  for (u32 i = 0; i < bottom_tensors_local_offset.size(); i++) {
    *(u32 *)p_ir_addr = bottom_tensors_local_offset[i];
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  *(u32 *)p_ir_addr = top_tensor_loc_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  if (p_ir_layer_info->extra_len > 0) {
    IR_USE_MEM_DATA(p_ir_addr, p_ir_layer_info->extra_buffer,
                    p_ir_layer_info->extra_len);
  }

  // output consumer number info
  *(u32 *)p_ir_addr = top_consumer_num;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_active_irbuf_write(void *p_ir_buf,
                                    ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  u32 bottom_tensor_id;
  u32 top_tensor_id;
  u32 top_consumer_num = 1;

  fw_active_layer_param_t fw_active_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_active_layer_param;

  *(fw_active_layer_param_t *)p_ir_addr = fw_active_layer_param;
  p_ir_addr = (fw_active_layer_param_t *)p_ir_addr + 1;

  bottom_tensor_id = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  top_tensor_id = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;
  top_consumer_num = p_ir_layer_info->ir_tensor_info_v[1].consumer_number;

  *(u32 *)p_ir_addr = (bottom_tensor_id << 16) | (top_tensor_id & 0xffff);
  p_ir_addr = (u32 *)p_ir_addr + 1;

  for (size_t i = 0; i < p_ir_layer_info->ir_tensor_info_v.size(); i++) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[i].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  // output consumer number info
  *(u32 *)p_ir_addr = top_consumer_num;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_crop_irbuf_write(void *p_ir_buf,
                                  ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  u32 bottom_tensor_id;
  u32 top_tensor_id;

  fw_crop_layer_param_t fw_crop_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_crop_layer_param;

  *(fw_crop_layer_param_t *)p_ir_addr = fw_crop_layer_param;
  p_ir_addr = (fw_crop_layer_param_t *)p_ir_addr + 1;

  bottom_tensor_id = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  top_tensor_id = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;

  *(u32 *)p_ir_addr = (bottom_tensor_id << 16) | (top_tensor_id & 0xffff);
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // output consumer number info
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].consumer_number;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_normalize_irbuf_write(void *p_ir_buf,
                                       ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  u32 bottom_tensor_id;
  u32 top_tensor_id;
  u32 top_consumer_num = 1;

  fw_normalize_layer_param_t fw_normalize_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_normalize_layer_param;

  *(fw_normalize_layer_param_t *)p_ir_addr = fw_normalize_layer_param;
  p_ir_addr = (fw_normalize_layer_param_t *)p_ir_addr + 1;

  bottom_tensor_id = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  if (fw_normalize_layer_param.channel_shared) {
    top_tensor_id = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;
    top_consumer_num = p_ir_layer_info->ir_tensor_info_v[1].consumer_number;
  } else {
    top_tensor_id = p_ir_layer_info->ir_tensor_info_v[2].tensor_id;
    top_consumer_num = p_ir_layer_info->ir_tensor_info_v[2].consumer_number;
  }
  *(u32 *)p_ir_addr = (bottom_tensor_id << 16) | (top_tensor_id & 0xffff);
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  if (fw_normalize_layer_param.channel_shared == 0) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[3].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  // output consumer number info
  *(u32 *)p_ir_addr = top_consumer_num;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *
static_loc_broadcast_binary_irbuf_write(void *p_ir_buf,
                                        ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_broadcast_binary_layer_param_t fw_broadcast_binary_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_broadcast_binary_layer_param;

  *(fw_broadcast_binary_layer_param_t *)p_ir_addr =
      fw_broadcast_binary_layer_param;
  p_ir_addr = (fw_broadcast_binary_layer_param_t *)p_ir_addr + 1;
  // input
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // output
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  if (p_ir_layer_info->ir_tensor_info_v.size() > 3) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[3].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }
  // output consumer number info
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].consumer_number;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_eltwise_binary_irbuf_write(void *p_ir_buf,
                                            ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_eltwise_binary_layer_param_t fw_eltwise_binary_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_eltwise_binary_layer_param;

  *(fw_eltwise_binary_layer_param_t *)p_ir_addr = fw_eltwise_binary_layer_param;
  p_ir_addr = (fw_eltwise_binary_layer_param_t *)p_ir_addr + 1;
  // input
  if (!fw_eltwise_binary_layer_param.a_is_coeff) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  if (!fw_eltwise_binary_layer_param.b_is_coeff) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // output
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  if (p_ir_layer_info->ir_tensor_info_v.size() > 3) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[3].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }
  // output consumer number info
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].consumer_number;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_const_binary_irbuf_write(void *p_ir_buf,
                                          ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_const_binary_layer_param_t fw_const_binary_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_const_binary_layer_param;

  *(fw_const_binary_layer_param_t *)p_ir_addr = fw_const_binary_layer_param;
  p_ir_addr = (fw_const_binary_layer_param_t *)p_ir_addr + 1;
  // input
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  // output
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  if (p_ir_layer_info->ir_tensor_info_v.size() > 2) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }
  // output consumer number info
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].consumer_number;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_strideslice_irbuf_write(void *p_ir_buf,
                                         ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  fw_stride_slice_layer_param_t fw_stride_slice_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_stride_slice_layer_param;

  *(fw_stride_slice_layer_param_t *)p_ir_addr = fw_stride_slice_layer_param;
  p_ir_addr = (fw_stride_slice_layer_param_t *)p_ir_addr + 1;
  // input
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  // output
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  // output consumer number info
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].consumer_number;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_reorg_irbuf_write(void *p_ir_buf,
                                   ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  u32 bottom_tensor_id;
  u32 top_tensor_id;

  fw_reorg_layer_param_t fw_reorg_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_reorg_layer_param;

  *(fw_reorg_layer_param_t *)p_ir_addr = fw_reorg_layer_param;
  p_ir_addr = (fw_reorg_layer_param_t *)p_ir_addr + 1;

  bottom_tensor_id = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  top_tensor_id = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;

  *(u32 *)p_ir_addr = (bottom_tensor_id << 16) | (top_tensor_id & 0xffff);
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // output consumer number info
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].consumer_number;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_dtype_convert_irbuf_write(void *p_ir_buf,
                                           ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_dtype_convert_layer_param_t fw_dtype_convert_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_dtype_convert_layer_param;

  *(fw_dtype_convert_layer_param_t *)p_ir_addr = fw_dtype_convert_layer_param;
  p_ir_addr = (fw_dtype_convert_layer_param_t *)p_ir_addr + 1;

  // input
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // output
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // imm buffer
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // output consumer number info
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].consumer_number;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_tile_irbuf_write(void *p_ir_buf,
                                  ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_tile_layer_param_t fw_tile_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_tile_layer_param;

  *(fw_tile_layer_param_t *)p_ir_addr = fw_tile_layer_param;
  p_ir_addr = (fw_tile_layer_param_t *)p_ir_addr + 1;

  // tensor_id
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // offset
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // tile_coeff tensor id
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // output consumer number info
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].consumer_number;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_pad_irbuf_write(void *p_ir_buf,
                                 ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_pad_layer_param_t fw_pad_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_pad_layer_param;

  *(fw_pad_layer_param_t *)p_ir_addr = fw_pad_layer_param;
  p_ir_addr = (fw_pad_layer_param_t *)p_ir_addr + 1;

  // tensor_id
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // offset
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // pad tensor id
  if (p_ir_layer_info->ir_tensor_info_v.size() > 2) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].tensor_id;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  // output consumer number info
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].consumer_number;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_lut_irbuf_write(void *p_ir_buf,
                                 ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  fw_lut_layer_param_t fw_lut_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_lut_layer_param;
  *(fw_lut_layer_param_t *)p_ir_addr = fw_lut_layer_param;
  p_ir_addr = (fw_lut_layer_param_t *)p_ir_addr + 1;
  // tensor_id
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  // mem offset
  *(u32 *)p_ir_addr =
      p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset; // input[0]
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr =
      p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset; // input[1]
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr =
      p_ir_layer_info->ir_tensor_info_v[2].local_mem_offset; // output[0]
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr =
      p_ir_layer_info->ir_tensor_info_v[3].local_mem_offset; // immbuf
  p_ir_addr = (u32 *)p_ir_addr + 1;

  // output consumer number info
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].consumer_number;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_common_loc_irbuf_write(void *buf, ir_layer_info_t *ir_layer_info,
                                    const void *param, int param_len) {
  void *p_ir_addr = buf;
  if (param != NULL) {
    IR_USE_MEM_DATA(p_ir_addr, param, param_len);
  }
  if (ir_layer_info->extra_len > 0) {
    assert((ir_layer_info->extra_len >> 24) == 0);
    int ver_len =
        (ir_layer_info->extra_version << 24) + ir_layer_info->extra_len;
    IR_USE_U32_DATA(p_ir_addr, ver_len); // keep compatible
    IR_USE_MEM_DATA(p_ir_addr, ir_layer_info->extra_buffer,
                    ir_layer_info->extra_len);
    // free the buffer mem
    // cannot free it in destruct function of ir_layer_info_t
    delete[] ir_layer_info->extra_buffer;
  }
  for (auto &tensor : ir_layer_info->ir_tensor_info_v) {
    IR_USE_LOC_TENSOR(p_ir_addr, tensor);
  }
  return p_ir_addr;
}

#define IMPLEMENT_LOC_IRBUF_WRITE(name)                                        \
  void *static_loc_##name##_irbuf_write(void *p_ir_buf,                        \
                                        ir_layer_info_t *ir_layer_info) {      \
    auto &layer_param =                                                        \
        ir_layer_info->fw_layer_param_u.fw_##name##_layer_param;               \
    return static_common_loc_irbuf_write(p_ir_buf, ir_layer_info,              \
                                         &layer_param, sizeof(layer_param));   \
  }

IMPLEMENT_LOC_IRBUF_WRITE(biasadd)
IMPLEMENT_LOC_IRBUF_WRITE(select)
IMPLEMENT_LOC_IRBUF_WRITE(upsample)
IMPLEMENT_LOC_IRBUF_WRITE(reduce_full)
IMPLEMENT_LOC_IRBUF_WRITE(interleave)
IMPLEMENT_LOC_IRBUF_WRITE(binary_shift)
IMPLEMENT_LOC_IRBUF_WRITE(arg)
IMPLEMENT_LOC_IRBUF_WRITE(conv3d)
IMPLEMENT_LOC_IRBUF_WRITE(pool3d)
IMPLEMENT_LOC_IRBUF_WRITE(group_norm)

void *static_loc_stridecalc_irbuf_write(void *p_ir_buf,
                                        ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_stridecalc_layer_param_t fw_stridecalc_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_stridecalc_layer_param;

  *(fw_stridecalc_layer_param_t *)p_ir_addr = fw_stridecalc_layer_param;
  p_ir_addr = (fw_stridecalc_layer_param_t *)p_ir_addr + 1;

  int A_is_const = (fw_stridecalc_layer_param.input_info >> 16) & 0xff;
  int B_is_const = (fw_stridecalc_layer_param.input_info >> 24) & 0xff;
  int input_num = 1;
  if (!A_is_const && !B_is_const && fw_stridecalc_layer_param.op_code != 5)
    input_num = 2;
  // input
  for (int i = 0; i < input_num; ++i) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[i].tensor_id;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[i].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  // output
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[input_num].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr =
      p_ir_layer_info->ir_tensor_info_v[input_num].local_mem_offset;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  if ((int)p_ir_layer_info->ir_tensor_info_v.size() > input_num + 1) {
    *(u32 *)p_ir_addr =
        p_ir_layer_info->ir_tensor_info_v[input_num + 1].local_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }
  // output consumer number info
  *(u32 *)p_ir_addr =
      p_ir_layer_info->ir_tensor_info_v[input_num].consumer_number;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_bitwise_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *p_ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  u32 top_consumer_num = 1;
  // layer_param
  fw_bitwise_layer_param_t fw_bitwise_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_bitwise_layer_param;
  *(fw_bitwise_layer_param_t *)p_ir_addr = fw_bitwise_layer_param;
  p_ir_addr = (fw_bitwise_layer_param_t *)p_ir_addr + 1;
  // tensor_id
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  if (fw_bitwise_layer_param.if_const == 0) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].tensor_id;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  } else {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }
  // mem offset
  *(u32 *)p_ir_addr =
      p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset; // input[0]
  p_ir_addr = (u32 *)p_ir_addr + 1;
  if (fw_bitwise_layer_param.if_const == 0) {
    *(u32 *)p_ir_addr =
        p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset; // input[1]
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(u32 *)p_ir_addr =
        p_ir_layer_info->ir_tensor_info_v[2].local_mem_offset; // output[0]
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(u32 *)p_ir_addr =
        p_ir_layer_info->ir_tensor_info_v[3].local_mem_offset; // immbuf
    p_ir_addr = (u32 *)p_ir_addr + 1;

    top_consumer_num = p_ir_layer_info->ir_tensor_info_v[2].consumer_number;
  } else {
    *(u32 *)p_ir_addr =
        p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset; // output[0]
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(u32 *)p_ir_addr =
        p_ir_layer_info->ir_tensor_info_v[2].local_mem_offset; // immbuf
    p_ir_addr = (u32 *)p_ir_addr + 1;

    top_consumer_num = p_ir_layer_info->ir_tensor_info_v[1].consumer_number;
  }
  // output consumer number info
  *(u32 *)p_ir_addr = top_consumer_num;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *
static_loc_arith_shift_irbuf_write(void *p_ir_buf, // input and output
                                   ir_layer_info_t *p_ir_layer_info) // input
{
  void *p_ir_addr = p_ir_buf;
  u32 top_consumer_num = 1;
  // layer param
  fw_arith_shift_layer_param_t fw_arith_shift_layer_param =
      p_ir_layer_info->fw_layer_param_u.fw_arith_shift_layer_param;
  *(fw_arith_shift_layer_param_t *)p_ir_addr = fw_arith_shift_layer_param;
  p_ir_addr = (fw_arith_shift_layer_param_t *)p_ir_addr + 1;
  // tensor_id
  *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[0].tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  if (fw_arith_shift_layer_param.shift_is_const == 0) {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[2].tensor_id;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  } else {
    *(u32 *)p_ir_addr = p_ir_layer_info->ir_tensor_info_v[1].tensor_id;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }
  // mem offset
  *(u32 *)p_ir_addr =
      p_ir_layer_info->ir_tensor_info_v[0].local_mem_offset; // input[0]
  p_ir_addr = (u32 *)p_ir_addr + 1;
  if (fw_arith_shift_layer_param.shift_is_const == 0) {
    *(u32 *)p_ir_addr =
        p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset; // input[1]
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(u32 *)p_ir_addr =
        p_ir_layer_info->ir_tensor_info_v[2].local_mem_offset; // output[0]
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(u32 *)p_ir_addr =
        p_ir_layer_info->ir_tensor_info_v[3].local_mem_offset; // immbuf
    p_ir_addr = (u32 *)p_ir_addr + 1;

    top_consumer_num = p_ir_layer_info->ir_tensor_info_v[2].consumer_number;
  } else {
    *(u32 *)p_ir_addr =
        p_ir_layer_info->ir_tensor_info_v[1].local_mem_offset; // output[0]
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(u32 *)p_ir_addr =
        p_ir_layer_info->ir_tensor_info_v[2].local_mem_offset; // immbuf
    p_ir_addr = (u32 *)p_ir_addr + 1;

    top_consumer_num = p_ir_layer_info->ir_tensor_info_v[1].consumer_number;
  }
  // output consumer number info
  *(u32 *)p_ir_addr = top_consumer_num;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  return p_ir_addr;
}

void *static_loc_tpu_irbuf_write(void *p_ir_addr,
                                 ir_layer_info_t *p_ir_layer_info) {
  auto &fw_param = p_ir_layer_info->fw_layer_param_u.fw_tpu_layer_param;
  *(fw_tpu_layer_param_t *)p_ir_addr = fw_param;
  p_ir_addr = (fw_tpu_layer_param_t *)p_ir_addr + 1;

  IR_USE_U32_DATA(p_ir_addr, p_ir_layer_info->extra_len);
  if (p_ir_layer_info->extra_len > 0) {
    IR_USE_MEM_DATA(p_ir_addr, p_ir_layer_info->extra_buffer,
                    p_ir_layer_info->extra_len);
  }
  // tensor_id
  for (auto tensor : p_ir_layer_info->ir_tensor_info_v) {
    IR_USE_U32_DATA(p_ir_addr, tensor.tensor_id);
    IR_USE_U32_DATA(p_ir_addr, tensor.local_mem_offset);
    if (tensor.is_io_tensor == 2) {
      IR_USE_U32_DATA(p_ir_addr, tensor.consumer_number);
    }
  }
  return p_ir_addr;
}

} // namespace tpu
} // namespace tpu_mlir

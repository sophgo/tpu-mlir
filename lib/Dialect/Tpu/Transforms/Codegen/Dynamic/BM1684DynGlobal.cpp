//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/BM1684DynGlobal.hpp"
#define IR_PACKING
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/BM1684DynIrUtils.hpp"

using namespace std;

namespace tpu_mlir {
namespace tpu {

void *call_global_layer_ir_write(FW_LAYER_TYPE_T fw_layer_type, void *p_ir_addr,
                                 ir_layer_info_t *ir_layer_info) {
#define GLB_IRBUF_CASE_BEGIN() if (0) {
#define GLB_IRBUF_CASE_END()                                                   \
  }                                                                            \
  else {                                                                       \
    std::cout << "not supported fw glb layer type " << fw_layer_type           \
              << " when writing global layer buffer" << std::endl;             \
    llvm_unreachable("fw_layer_type error");                                   \
  }
// Function name is different to layer paramter type
#define GLB_IRBUF_CASE_BY_PARAM(TYPE, name, param_t)                           \
  }                                                                            \
  else if (fw_layer_type == FW_BMNET_##TYPE) {                                 \
    *(u32 *)p_ir_addr = (u32)(sizeof(param_t));                                \
    p_ir_addr = (u32 *)p_ir_addr + 1;                                          \
    p_ir_addr = static_glb_##name##_irbuf_write(p_ir_addr, ir_layer_info);

// Function name is same as layer parameter
#define GLB_IRBUF_CASE(TYPE, name)                                             \
  GLB_IRBUF_CASE_BY_PARAM(TYPE, name, fw_##name##_layer_param_t)

// No layer parameter
#define GLB_IRBUF_CASE_NO_PARAM(TYPE, name)                                    \
  }                                                                            \
  else if (fw_layer_type == FW_BMNET_##TYPE) {                                 \
    *(u32 *)p_ir_addr = 0;                                                     \
    p_ir_addr = (u32 *)p_ir_addr + 1;                                          \
    p_ir_addr = static_glb_##name##_irbuf_write(p_ir_addr, ir_layer_info);

  GLB_IRBUF_CASE_BEGIN()
  GLB_IRBUF_CASE(FC, fc)
  GLB_IRBUF_CASE(BATCHNORM, batchnorm)
  GLB_IRBUF_CASE(SCALE, scale)
  GLB_IRBUF_CASE(ELTWISE, eltwise)
  GLB_IRBUF_CASE(SOFTMAX, softmax)
  GLB_IRBUF_CASE(RELU, prelu)
  GLB_IRBUF_CASE(PRELU, prelu)
  GLB_IRBUF_CASE(CONV, conv)
  GLB_IRBUF_CASE_BY_PARAM(POOL, pooling, fw_pool_layer_param_t)
  GLB_IRBUF_CASE(RPN, rpn)
  GLB_IRBUF_CASE(RESHAPE, reshape)
  GLB_IRBUF_CASE(CROP, crop)
  GLB_IRBUF_CASE(PERMUTE, permute)
  GLB_IRBUF_CASE_BY_PARAM(POOLTF, pooling_tf, fw_pooltf_layer_param_t)
  GLB_IRBUF_CASE(SPLIT_TF, split_tf)
  GLB_IRBUF_CASE(DECONV, deconv)
  GLB_IRBUF_CASE(PAD, pad)
  GLB_IRBUF_CASE(ARG, arg)
  GLB_IRBUF_CASE(TILE, tile)
  GLB_IRBUF_CASE(REDUCE, reduce)
  GLB_IRBUF_CASE(SELECT, select)
  GLB_IRBUF_CASE(WHERE, where)
  GLB_IRBUF_CASE(MASKED_SELECT, masked_select)
  GLB_IRBUF_CASE(INDEX_SELECT, index_select)
  GLB_IRBUF_CASE(NMS, nms)
  GLB_IRBUF_CASE(BROADCAST_BINARY, broadcast_binary)
  GLB_IRBUF_CASE(CONST_BINARY, const_binary)
  GLB_IRBUF_CASE(ELTWISE_BINARY, eltwise_binary)
  GLB_IRBUF_CASE(BIASADD, biasadd)
  GLB_IRBUF_CASE(ACTIVE, active)
  GLB_IRBUF_CASE(NORMALIZE, normalize)
  GLB_IRBUF_CASE(SHAPE_CONST, shape_const)
  GLB_IRBUF_CASE(SHAPE_REF, shape_ref)
  GLB_IRBUF_CASE(SHAPE_OP, shape_op)
  GLB_IRBUF_CASE(SHAPE_SLICE, shape_slice)
  GLB_IRBUF_CASE(SHAPE_PACK, shape_pack)
  GLB_IRBUF_CASE_NO_PARAM(SHAPE_ASSIGN, shape_assign)
  GLB_IRBUF_CASE(SHAPE_REORDER, shape_reorder)
  GLB_IRBUF_CASE(EXPAND_DIM, expand_dims)
  GLB_IRBUF_CASE(SQUEEZE_DIM, squeeze)
  GLB_IRBUF_CASE(REF_PAD, ref_pad)
  GLB_IRBUF_CASE_NO_PARAM(REF_CROP, ref_crop)
  GLB_IRBUF_CASE(TRANSPOSE, transpose)
  GLB_IRBUF_CASE(REDUCE_FULL, reduce_full)
  GLB_IRBUF_CASE(CONCAT, concat)
  GLB_IRBUF_CASE(STRIDESLICE, stride_slice)
  GLB_IRBUF_CASE(BATCH2SPACE, batch2space)
  GLB_IRBUF_CASE(SPACE2BATCH, space2batch)
  GLB_IRBUF_CASE_NO_PARAM(OUTPUT, output)
  GLB_IRBUF_CASE(INTERP, interp)
  GLB_IRBUF_CASE(EXPAND, expand)
  GLB_IRBUF_CASE(EMBEDDING, embedding)
  GLB_IRBUF_CASE(TOPK, topk)
  GLB_IRBUF_CASE(CUMSUM, cumsum)
  GLB_IRBUF_CASE(SHAPE_ADDN, shape_addn)
  GLB_IRBUF_CASE(ROIPOOLING, roi_pooling)
  GLB_IRBUF_CASE(PSROIPOOLING, psroi_pooling)
  GLB_IRBUF_CASE(LRN, lrn)
  GLB_IRBUF_CASE(CONSTANT_FILL, constant_fill)
  GLB_IRBUF_CASE(SIMPLE_CROP, simple_crop)
  GLB_IRBUF_CASE(SLICELIKE, slicelike)
  GLB_IRBUF_CASE(ADAPTIVEPOOLING, adaptive_pool)
  GLB_IRBUF_CASE(BATCH_MATMUL, batch_matmul)
  GLB_IRBUF_CASE_NO_PARAM(SHAPE_RANGE, shape_range)
  GLB_IRBUF_CASE(SHAPE_TILE, shape_tile)
  GLB_IRBUF_CASE(SHAPE_REVERSE, shape_reverse)
  GLB_IRBUF_CASE(SHAPE_EXPAND_NDIMS, shape_expand_ndims)
  GLB_IRBUF_CASE(SHAPE_CAST, shape_cast)
  GLB_IRBUF_CASE_NO_PARAM(SHAPE_RESHAPE, shape_reshape)
  GLB_IRBUF_CASE(SHAPE_REDUCE, shape_reduce)
  GLB_IRBUF_CASE(DTYPE_CONVERT, dtype_convert)
  GLB_IRBUF_CASE(PRIORBOX, priorbox)
  GLB_IRBUF_CASE(MULSHIFT, mulshift)
  GLB_IRBUF_CASE(YOLO, yolo)
  GLB_IRBUF_CASE(YOLOV3_DETECT_OUT, yolov3_detect_out)
  GLB_IRBUF_CASE(SSD_DETECT_OUT, ssd_detect_out)
  GLB_IRBUF_CASE_NO_PARAM(DEVICE2HOST, device2host)
  GLB_IRBUF_CASE_NO_PARAM(HOST2DEVICE, host2device)
  GLB_IRBUF_CASE(TENSOR_ARRAY, tensorarray)
  GLB_IRBUF_CASE_NO_PARAM(TA_SIZE, ta_size)
  GLB_IRBUF_CASE_NO_PARAM(TA_READ, ta_read)
  GLB_IRBUF_CASE_BY_PARAM(TA_WRITE, ta_write, fw_tensor_array_op_param_t)
  GLB_IRBUF_CASE_BY_PARAM(TA_SCATTER, ta_scatter, fw_tensor_array_op_param_t)
  GLB_IRBUF_CASE_NO_PARAM(TA_GATHER, ta_gather)
  GLB_IRBUF_CASE_BY_PARAM(TA_SPLIT, ta_split, fw_tensor_array_op_param_t)
  GLB_IRBUF_CASE_NO_PARAM(TA_CONCAT, ta_concat)
  GLB_IRBUF_CASE(SHAPE_SPLIT, shape_split)
  GLB_IRBUF_CASE(SHAPE_UNARY, shape_unary)
  GLB_IRBUF_CASE(SHAPE_SQUEEZE, shape_squeeze)
  GLB_IRBUF_CASE(RANK, rank)
  GLB_IRBUF_CASE(SLICE, slice)
  GLB_IRBUF_CASE(SHAPE_SIZESLICE, shape_sizeslice)
  GLB_IRBUF_CASE(COEFF2NEURON, coeff2neuron)
  GLB_IRBUF_CASE(SHAPE_SELECT, shape_select)
  GLB_IRBUF_CASE(DEPTH2SPACE, depth2space)
  GLB_IRBUF_CASE(WHERE_SQUEEZE_GATHER, where_squeeze_gather)
  GLB_IRBUF_CASE(SORT_PER_DIM, sort_per_dim)
  GLB_IRBUF_CASE(REVERSE, reverse)
  GLB_IRBUF_CASE(LSTM, lstm)
  GLB_IRBUF_CASE(BROADCAST_LIKE, broadcast_like)
  GLB_IRBUF_CASE(REORG, reorg)
  GLB_IRBUF_CASE(LUT, lut)
  GLB_IRBUF_CASE(MATRIX_BAND_PART, matrix_band_part)
  GLB_IRBUF_CASE(CONV3D, conv3d)
  GLB_IRBUF_CASE(POOL3D, pool3d)
  GLB_IRBUF_CASE(STRIDECALC, stridecalc)
  GLB_IRBUF_CASE(INTERLEAVE, interleave)
  GLB_IRBUF_CASE(BITWISE, bitwise)
  GLB_IRBUF_CASE(BINARY_SHIFT, binary_shift)
  GLB_IRBUF_CASE(GRU, gru)
  GLB_IRBUF_CASE(PYTORCH_LSTM, pytorch_lstm)
  GLB_IRBUF_CASE(ARITH_SHIFT, arith_shift)
  GLB_IRBUF_CASE(MULTI_MASKED_SELECT, multi_masked_select)
  GLB_IRBUF_CASE(TPU, tpu)
  GLB_IRBUF_CASE_NO_PARAM(SEQUENCE_GEN, sequence_gen)
  GLB_IRBUF_CASE(UPSAMPLEMASK, upsample_mask)
  GLB_IRBUF_CASE(UPSAMPLE, upsample)
  GLB_IRBUF_CASE(GROUP_NORM, group_norm)
  GLB_IRBUF_CASE(DECONV3D, deconv3d)
  // global_dynamic: add layer_type irbuf_write case
  GLB_IRBUF_CASE_END()
  return p_ir_addr;
}

static void *write_tensor_info(void *p_ir_addr, ir_tensor_info_t *info) {
  u32 is_io_tensor = info->is_io_tensor;
  *(u32 *)p_ir_addr = is_io_tensor;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  *(u32 *)p_ir_addr = info->tensor_id;
  p_ir_addr = (u32 *)p_ir_addr + 1;
  if (!is_io_tensor) {
    *(u64 *)p_ir_addr = info->global_mem_offset;
    p_ir_addr = (u64 *)p_ir_addr + 1;
  }
  return p_ir_addr;
}

void *static_glb_rpn_irbuf_write(void *p_ir_buf,
                                 ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_rpn_layer_param_t fw_rpn_layer_param =
      ir_layer_info->fw_layer_param_u.fw_rpn_layer_param;

  *(fw_rpn_layer_param_t *)p_ir_addr = fw_rpn_layer_param;
  p_ir_addr = (fw_rpn_layer_param_t *)p_ir_addr + 1;

  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[1]);
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[2]);
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[3]);

  return p_ir_addr;
}

void *static_glb_fc_irbuf_write(void *p_ir_buf,
                                ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_fc_layer_param_t fw_fc_layer_param =
      ir_layer_info->fw_layer_param_u.fw_fc_layer_param;

  *(fw_fc_layer_param_t *)p_ir_addr = fw_fc_layer_param;
  p_ir_addr = (fw_fc_layer_param_t *)p_ir_addr + 1;

  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);

  if (fw_fc_layer_param.weight_is_datatensor) {
    p_ir_addr =
        write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[1]);
  } else {
    *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[1].global_mem_offset;
    p_ir_addr = (u64 *)p_ir_addr + 1;
  }
  if (fw_fc_layer_param.if_activated && fw_fc_layer_param.active_type == 2 &&
      !fw_fc_layer_param.channel_shared) {
    if (fw_fc_layer_param.using_bias == 1) {
      *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[2].global_mem_offset;
      p_ir_addr = (u64 *)p_ir_addr + 1;

      *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[3].global_mem_offset;
      p_ir_addr = (u64 *)p_ir_addr + 1;

      p_ir_addr =
          write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[4]);
    } else {

      p_ir_addr =
          write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[2]);
      *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[3].global_mem_offset;
      p_ir_addr = (u64 *)p_ir_addr + 1;
    }
  } else {
    if (fw_fc_layer_param.using_bias == 1) {
      *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[2].global_mem_offset;
      p_ir_addr = (u64 *)p_ir_addr + 1;
      p_ir_addr =
          write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[3]);
    } else {
      p_ir_addr =
          write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[2]);
    }
  }

  return p_ir_addr;
}

void *static_glb_batchnorm_irbuf_write(void *p_ir_buf,
                                       ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_batchnorm_layer_param_t fw_batchnorm_layer_param =
      ir_layer_info->fw_layer_param_u.fw_batchnorm_layer_param;

  *(fw_batchnorm_layer_param_t *)p_ir_addr = fw_batchnorm_layer_param;
  p_ir_addr = (fw_batchnorm_layer_param_t *)p_ir_addr + 1;

  // input
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);

  // mean_ma_global_offset
  *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[1].global_mem_offset;
  p_ir_addr = (u64 *)p_ir_addr + 1;

  // variance_ma_global_offset
  *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[2].global_mem_offset;
  p_ir_addr = (u64 *)p_ir_addr + 1;

  int non_scale_size = ir_layer_info->ir_tensor_info_v.size();
  if (fw_batchnorm_layer_param.if_scale && fw_batchnorm_layer_param.scale_bias)
    non_scale_size -= 2;
  else if (fw_batchnorm_layer_param.if_scale &&
           !fw_batchnorm_layer_param.scale_bias)
    non_scale_size -= 1;
  else
    non_scale_size = non_scale_size;

  int idx = 3;
  // rshift global offset
  if (non_scale_size > 4) {
    *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[idx].global_mem_offset;
    p_ir_addr = (u64 *)p_ir_addr + 1;
    idx++;
  }

  if (fw_batchnorm_layer_param.if_scale == 1) {
    *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[idx].global_mem_offset;
    p_ir_addr = (u64 *)p_ir_addr + 1;
    idx++;
    if (fw_batchnorm_layer_param.scale_bias == 1) {
      *(u64 *)p_ir_addr =
          ir_layer_info->ir_tensor_info_v[idx].global_mem_offset;
      p_ir_addr = (u64 *)p_ir_addr + 1;
      idx++;
    }
  }

  // output
  p_ir_addr =
      write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[idx]);
  return p_ir_addr;
}

void *static_glb_scale_irbuf_write(void *p_ir_buf,
                                   ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_scale_layer_param_t fw_scale_layer_param =
      ir_layer_info->fw_layer_param_u.fw_scale_layer_param;

  *(fw_scale_layer_param_t *)p_ir_addr = fw_scale_layer_param;
  p_ir_addr = (fw_scale_layer_param_t *)p_ir_addr + 1;

  // input
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);

  // scale_global_offset
  *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[1].global_mem_offset;
  p_ir_addr = (u64 *)p_ir_addr + 1;

  // bias_global_offset
  int using_bias = fw_scale_layer_param.using_bias;
  int idx = 2;
  if (using_bias) {
    *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[2].global_mem_offset;
    p_ir_addr = (u64 *)p_ir_addr + 1;
    idx++;
  }

  for (; idx < (int)ir_layer_info->ir_tensor_info_v.size() - 1;) {
    *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[idx].global_mem_offset;
    p_ir_addr = (u64 *)p_ir_addr + 1;
    idx++;
  }
  // output
  p_ir_addr =
      write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[idx]);

  return p_ir_addr;
}

void *static_glb_eltwise_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  u8 is_io_tensor;

  fw_eltwise_layer_param_t fw_eltwise_layer_param =
      ir_layer_info->fw_layer_param_u.fw_eltwise_layer_param;

  *(fw_eltwise_layer_param_t *)p_ir_addr = fw_eltwise_layer_param;
  p_ir_addr = (fw_eltwise_layer_param_t *)p_ir_addr + 1;

  // input
  u32 idx = 0;
  for (idx = 0; idx < fw_eltwise_layer_param.in_num; idx++) {
    is_io_tensor = ir_layer_info->ir_tensor_info_v[idx].is_io_tensor;
    *(u32 *)p_ir_addr = (u32)is_io_tensor;
    p_ir_addr = (u32 *)p_ir_addr + 1;

    *(u32 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[idx].tensor_id;
    p_ir_addr = (u32 *)p_ir_addr + 1;
    if (fw_eltwise_layer_param.op_code == 1) {
      *(float *)p_ir_addr = ir_layer_info->ir_tensor_info_v[idx].coeff;
      p_ir_addr = (float *)p_ir_addr + 1;
    }

    if (!is_io_tensor) {
      *(u64 *)p_ir_addr =
          ir_layer_info->ir_tensor_info_v[idx].global_mem_offset;
      p_ir_addr = (u64 *)p_ir_addr + 1;
    }
  }

  // output
  p_ir_addr = write_tensor_info(
      p_ir_addr,
      &ir_layer_info->ir_tensor_info_v[fw_eltwise_layer_param.in_num]);

  if (ir_layer_info->ir_tensor_info_v.size() >
      (u32)(fw_eltwise_layer_param.in_num + 1)) {
    *(u32 *)p_ir_addr =
        ir_layer_info->ir_tensor_info_v[fw_eltwise_layer_param.in_num + 1]
            .global_mem_offset;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  return p_ir_addr;
}

void *static_glb_softmax_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_softmax_layer_param_t fw_softmax_layer_param =
      ir_layer_info->fw_layer_param_u.fw_softmax_layer_param;

  *(fw_softmax_layer_param_t *)p_ir_addr = fw_softmax_layer_param;
  p_ir_addr = (fw_softmax_layer_param_t *)p_ir_addr + 1;

  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[1]);

  return p_ir_addr;
}

void *static_glb_prelu_irbuf_write(void *p_ir_buf,
                                   ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_prelu_layer_param_t fw_prelu_layer_param =
      ir_layer_info->fw_layer_param_u.fw_prelu_layer_param;

  *(fw_prelu_layer_param_t *)p_ir_addr = fw_prelu_layer_param;
  p_ir_addr = (fw_prelu_layer_param_t *)p_ir_addr + 1;

  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);

  if (fw_prelu_layer_param.channel_shared) {
    p_ir_addr =
        write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[1]);
  } else {
    *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[1].global_mem_offset;
    p_ir_addr = (u64 *)p_ir_addr + 1;
    p_ir_addr =
        write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[2]);
  }

  return p_ir_addr;
}

void *static_glb_pooling_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_pool_layer_param_t fw_pool_layer_param =
      ir_layer_info->fw_layer_param_u.fw_pool_layer_param;

  *(fw_pool_layer_param_t *)p_ir_addr = fw_pool_layer_param;
  p_ir_addr = (fw_pool_layer_param_t *)p_ir_addr + 1;

  // input
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);

  // output
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[1]);
  if (fw_pool_layer_param.is_avg_pool == 2) {
    p_ir_addr =
        write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[2]);
  }

  return p_ir_addr;
}

void *static_glb_pooling_tf_irbuf_write(void *p_ir_buf,
                                        ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_pooltf_layer_param_t fw_pooltf_layer_param =
      ir_layer_info->fw_layer_param_u.fw_pooltf_layer_param;

  *(fw_pooltf_layer_param_t *)p_ir_addr = fw_pooltf_layer_param;
  p_ir_addr = (fw_pooltf_layer_param_t *)p_ir_addr + 1;

  // input
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);

  // output
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[1]);

  return p_ir_addr;
}

void *static_glb_conv_irbuf_write(void *p_ir_buf,
                                  ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_conv_layer_param_t fw_conv_layer_param =
      ir_layer_info->fw_layer_param_u.fw_conv_layer_param;

  *(fw_conv_layer_param_t *)p_ir_addr = fw_conv_layer_param;
  p_ir_addr = (fw_conv_layer_param_t *)p_ir_addr + 1;

  int idx = 0;
  // input
  p_ir_addr =
      write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[idx]);
  idx++;

  // weight
  if (fw_conv_layer_param.weight_is_tensor) {
    p_ir_addr =
        write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[idx]);
  } else {
    *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[idx].global_mem_offset;
    p_ir_addr = (u64 *)p_ir_addr + 1;
  }
  idx++;

  // bias
  if (fw_conv_layer_param.using_bias == 1) {
    *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[idx].global_mem_offset;
    p_ir_addr = (u64 *)p_ir_addr + 1;
    idx++;
  }

  if (fw_conv_layer_param.if_batchnorm == 1) {
    *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[idx].global_mem_offset;
    p_ir_addr = (u64 *)p_ir_addr + 1;
    idx++;
    *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[idx].global_mem_offset;
    p_ir_addr = (u64 *)p_ir_addr + 1;
    idx++;
  }

  if (fw_conv_layer_param.if_scale == 1) {
    *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[idx].global_mem_offset;
    p_ir_addr = (u64 *)p_ir_addr + 1;
    idx++;
    if (fw_conv_layer_param.scale_bias == 1) {
      *(u64 *)p_ir_addr =
          ir_layer_info->ir_tensor_info_v[idx].global_mem_offset;
      p_ir_addr = (u64 *)p_ir_addr + 1;
      idx++;
    }
  }
  // output
  p_ir_addr =
      write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[idx]);

  return p_ir_addr;
}

void *static_glb_deconv_irbuf_write(void *p_ir_buf,
                                    ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_deconv_layer_param_t fw_deconv_layer_param =
      ir_layer_info->fw_layer_param_u.fw_deconv_layer_param;

  *(fw_deconv_layer_param_t *)p_ir_addr = fw_deconv_layer_param;
  p_ir_addr = (fw_deconv_layer_param_t *)p_ir_addr + 1;

  // input
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);

  // weight
  *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[1].global_mem_offset;
  p_ir_addr = (u64 *)p_ir_addr + 1;

  // bias
  int idx = 2;
  if (fw_deconv_layer_param.using_bias) {
    *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[idx].global_mem_offset;
    p_ir_addr = (u64 *)p_ir_addr + 1;
    idx++;
  }

  // output
  p_ir_addr =
      write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[idx]);
  return p_ir_addr;
}

void *static_glb_reshape_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_reshape_layer_param_t fw_reshape_layer_param =
      ir_layer_info->fw_layer_param_u.fw_reshape_layer_param;

  *(fw_reshape_layer_param_t *)p_ir_addr = fw_reshape_layer_param;
  p_ir_addr = (fw_reshape_layer_param_t *)p_ir_addr + 1;
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[1]);
  return p_ir_addr;
}

void *static_glb_permute_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_permute_layer_param_t fw_permute_layer_param =
      ir_layer_info->fw_layer_param_u.fw_permute_layer_param;
  *(fw_permute_layer_param_t *)p_ir_addr = fw_permute_layer_param;
  p_ir_addr = (fw_permute_layer_param_t *)p_ir_addr + 1;
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[1]);
  return p_ir_addr;
}

void *static_glb_crop_irbuf_write(void *p_ir_buf,
                                  ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_crop_layer_param_t fw_crop_layer_param =
      ir_layer_info->fw_layer_param_u.fw_crop_layer_param;

  *(fw_crop_layer_param_t *)p_ir_addr = fw_crop_layer_param;
  p_ir_addr = (fw_crop_layer_param_t *)p_ir_addr + 1;

  // input
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);

  // output
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[1]);

  return p_ir_addr;
}

void *static_glb_pad_irbuf_write(void *p_ir_buf,
                                 ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_pad_layer_param_t fw_pad_layer_param =
      ir_layer_info->fw_layer_param_u.fw_pad_layer_param;

  *(fw_pad_layer_param_t *)p_ir_addr = fw_pad_layer_param;
  p_ir_addr = (fw_pad_layer_param_t *)p_ir_addr + 1;

  for (auto &tensor : ir_layer_info->ir_tensor_info_v) {
    IR_USE_GLB_TENSOR(p_ir_addr, tensor);
  }

  return p_ir_addr;
}

void *static_glb_arg_irbuf_write(void *p_ir_buf,
                                 ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_arg_layer_param_t fw_arg_layer_param =
      ir_layer_info->fw_layer_param_u.fw_arg_layer_param;

  *(fw_arg_layer_param_t *)p_ir_addr = fw_arg_layer_param;
  p_ir_addr = (fw_arg_layer_param_t *)p_ir_addr + 1;

  // input
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);

  // output
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[1]);

  return p_ir_addr;
}

void *static_glb_active_irbuf_write(void *p_ir_buf,
                                    ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_active_layer_param_t fw_active_layer_param =
      ir_layer_info->fw_layer_param_u.fw_active_layer_param;

  *(fw_active_layer_param_t *)p_ir_addr = fw_active_layer_param;
  p_ir_addr = (fw_active_layer_param_t *)p_ir_addr + 1;

  // input
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);

  // output
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[1]);

  return p_ir_addr;
}

void *static_glb_normalize_irbuf_write(void *p_ir_buf,
                                       ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_normalize_layer_param_t fw_normalize_layer_param =
      ir_layer_info->fw_layer_param_u.fw_normalize_layer_param;

  *(fw_normalize_layer_param_t *)p_ir_addr = fw_normalize_layer_param;
  p_ir_addr = (fw_normalize_layer_param_t *)p_ir_addr + 1;

  int idx = 0;
  // input
  p_ir_addr =
      write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[idx]);
  idx++;

  if (!fw_normalize_layer_param.channel_shared) {
    *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[idx].global_mem_offset;
    p_ir_addr = (u64 *)p_ir_addr + 1;
    idx++;
  }

  p_ir_addr =
      write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[idx]);

  return p_ir_addr;
}

void *static_common_glb_irbuf_write(void *buf, ir_layer_info_t *ir_layer_info,
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
    IR_USE_GLB_TENSOR(p_ir_addr, tensor);
  }
  return p_ir_addr;
}
void *static_glb_priorbox_irbuf_write(void *p_ir_buf,
                                      ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  fw_priorbox_layer_param_t fw_priorbox_layer_param =
      ir_layer_info->fw_layer_param_u.fw_priorbox_layer_param;

  *(fw_priorbox_layer_param_t *)p_ir_addr = fw_priorbox_layer_param;
  p_ir_addr = (fw_priorbox_layer_param_t *)p_ir_addr + 1;

  // input
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[1]);

  // output
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[2]);

  return p_ir_addr;
}

void *static_glb_mulshift_irbuf_write(void *p_ir_buf,
                                      ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_mulshift_layer_param_t fw_mulshift_layer_param =
      ir_layer_info->fw_layer_param_u.fw_mulshift_layer_param;

  *(fw_mulshift_layer_param_t *)p_ir_addr = fw_mulshift_layer_param;
  p_ir_addr = (fw_mulshift_layer_param_t *)p_ir_addr + 1;

  // input
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);

  // output
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[1]);

  return p_ir_addr;
}

void *static_glb_yolo_irbuf_write(void *p_ir_buf,
                                  ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;

  fw_yolo_layer_param_t fw_yolo_layer_param =
      ir_layer_info->fw_layer_param_u.fw_yolo_layer_param;

  *(fw_yolo_layer_param_t *)p_ir_addr = fw_yolo_layer_param;
  p_ir_addr = (fw_yolo_layer_param_t *)p_ir_addr + 1;

  // input
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);

  // output
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[1]);

  return p_ir_addr;
}

void *static_glb_sequence_gen_irbuf_write(void *p_ir_buf,
                                          ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  for (auto &tensor : ir_layer_info->ir_tensor_info_v) {
    IR_USE_GLB_TENSOR(p_ir_addr, tensor);
  }
  return p_ir_addr;
}

#define IMPLEMENT_GLB_IRBUF_WRITE_PARAM(name, param_name)                      \
  void *static_glb_##name##_irbuf_write(void *p_ir_buf,                        \
                                        ir_layer_info_t *ir_layer_info) {      \
    auto &layer_param = ir_layer_info->fw_layer_param_u.param_name;            \
    return static_common_glb_irbuf_write(p_ir_buf, ir_layer_info,              \
                                         &layer_param, sizeof(layer_param));   \
  }
#define IMPLEMENT_GLB_IRBUF_WRITE(name)                                        \
  IMPLEMENT_GLB_IRBUF_WRITE_PARAM(name, fw_##name##_layer_param)

#define IMPLEMENT_GLB_IRBUF_WRITE_NO_PARAM(name)                               \
  void *static_glb_##name##_irbuf_write(void *p_ir_buf,                        \
                                        ir_layer_info_t *ir_layer_info) {      \
    return static_common_glb_irbuf_write(p_ir_buf, ir_layer_info, NULL, 0);    \
  }

IMPLEMENT_GLB_IRBUF_WRITE(tile)
IMPLEMENT_GLB_IRBUF_WRITE(reduce)
IMPLEMENT_GLB_IRBUF_WRITE(select)
IMPLEMENT_GLB_IRBUF_WRITE(where)
IMPLEMENT_GLB_IRBUF_WRITE(masked_select)
IMPLEMENT_GLB_IRBUF_WRITE(multi_masked_select)
IMPLEMENT_GLB_IRBUF_WRITE(index_select)
IMPLEMENT_GLB_IRBUF_WRITE(sort_per_dim)
IMPLEMENT_GLB_IRBUF_WRITE(nms)
IMPLEMENT_GLB_IRBUF_WRITE(broadcast_binary)
IMPLEMENT_GLB_IRBUF_WRITE(eltwise_binary)
IMPLEMENT_GLB_IRBUF_WRITE(const_binary)
IMPLEMENT_GLB_IRBUF_WRITE(biasadd)
IMPLEMENT_GLB_IRBUF_WRITE(where_squeeze_gather)

IMPLEMENT_GLB_IRBUF_WRITE(shape_const)
IMPLEMENT_GLB_IRBUF_WRITE(shape_ref)
IMPLEMENT_GLB_IRBUF_WRITE(shape_op)
IMPLEMENT_GLB_IRBUF_WRITE(shape_slice)
IMPLEMENT_GLB_IRBUF_WRITE(shape_pack)
IMPLEMENT_GLB_IRBUF_WRITE_NO_PARAM(shape_assign)
IMPLEMENT_GLB_IRBUF_WRITE(shape_reorder)
IMPLEMENT_GLB_IRBUF_WRITE(expand_dims)
IMPLEMENT_GLB_IRBUF_WRITE(squeeze)
IMPLEMENT_GLB_IRBUF_WRITE(ref_pad)
IMPLEMENT_GLB_IRBUF_WRITE_NO_PARAM(ref_crop)
IMPLEMENT_GLB_IRBUF_WRITE(transpose)
IMPLEMENT_GLB_IRBUF_WRITE(reduce_full)
IMPLEMENT_GLB_IRBUF_WRITE(concat)
IMPLEMENT_GLB_IRBUF_WRITE(stride_slice)
IMPLEMENT_GLB_IRBUF_WRITE(split_tf)
IMPLEMENT_GLB_IRBUF_WRITE(batch2space)
IMPLEMENT_GLB_IRBUF_WRITE(space2batch)
IMPLEMENT_GLB_IRBUF_WRITE_NO_PARAM(output)
IMPLEMENT_GLB_IRBUF_WRITE(interp)
IMPLEMENT_GLB_IRBUF_WRITE(expand)
IMPLEMENT_GLB_IRBUF_WRITE(embedding)
IMPLEMENT_GLB_IRBUF_WRITE(topk)
IMPLEMENT_GLB_IRBUF_WRITE(cumsum)
IMPLEMENT_GLB_IRBUF_WRITE(shape_addn)
IMPLEMENT_GLB_IRBUF_WRITE(roi_pooling)
IMPLEMENT_GLB_IRBUF_WRITE(psroi_pooling)
IMPLEMENT_GLB_IRBUF_WRITE(lrn)
IMPLEMENT_GLB_IRBUF_WRITE(constant_fill)
IMPLEMENT_GLB_IRBUF_WRITE(simple_crop)
IMPLEMENT_GLB_IRBUF_WRITE(slicelike)
IMPLEMENT_GLB_IRBUF_WRITE(adaptive_pool)
IMPLEMENT_GLB_IRBUF_WRITE(batch_matmul)
IMPLEMENT_GLB_IRBUF_WRITE_NO_PARAM(shape_range)
IMPLEMENT_GLB_IRBUF_WRITE(shape_tile)
IMPLEMENT_GLB_IRBUF_WRITE(shape_reverse)
IMPLEMENT_GLB_IRBUF_WRITE(shape_expand_ndims)
IMPLEMENT_GLB_IRBUF_WRITE(shape_cast)
IMPLEMENT_GLB_IRBUF_WRITE_NO_PARAM(shape_reshape)
IMPLEMENT_GLB_IRBUF_WRITE(shape_reduce)
IMPLEMENT_GLB_IRBUF_WRITE(dtype_convert)
IMPLEMENT_GLB_IRBUF_WRITE(ssd_detect_out)
IMPLEMENT_GLB_IRBUF_WRITE(yolov3_detect_out)
IMPLEMENT_GLB_IRBUF_WRITE_NO_PARAM(host2device)
IMPLEMENT_GLB_IRBUF_WRITE_NO_PARAM(device2host)
IMPLEMENT_GLB_IRBUF_WRITE(tensorarray)
IMPLEMENT_GLB_IRBUF_WRITE_NO_PARAM(ta_size)
IMPLEMENT_GLB_IRBUF_WRITE_NO_PARAM(ta_read)
IMPLEMENT_GLB_IRBUF_WRITE_NO_PARAM(ta_gather)
IMPLEMENT_GLB_IRBUF_WRITE_NO_PARAM(ta_concat)
IMPLEMENT_GLB_IRBUF_WRITE_PARAM(ta_write, fw_tensor_array_op_param)
IMPLEMENT_GLB_IRBUF_WRITE_PARAM(ta_scatter, fw_tensor_array_op_param)
IMPLEMENT_GLB_IRBUF_WRITE_PARAM(ta_split, fw_tensor_array_op_param)
IMPLEMENT_GLB_IRBUF_WRITE(shape_split)
IMPLEMENT_GLB_IRBUF_WRITE(shape_unary)
IMPLEMENT_GLB_IRBUF_WRITE(shape_squeeze)
IMPLEMENT_GLB_IRBUF_WRITE(rank)
IMPLEMENT_GLB_IRBUF_WRITE(slice)
IMPLEMENT_GLB_IRBUF_WRITE(shape_sizeslice)
IMPLEMENT_GLB_IRBUF_WRITE(coeff2neuron)
IMPLEMENT_GLB_IRBUF_WRITE(shape_select)
IMPLEMENT_GLB_IRBUF_WRITE(depth2space)
IMPLEMENT_GLB_IRBUF_WRITE(reverse)
IMPLEMENT_GLB_IRBUF_WRITE(lstm)
IMPLEMENT_GLB_IRBUF_WRITE(broadcast_like)
IMPLEMENT_GLB_IRBUF_WRITE(reorg)
IMPLEMENT_GLB_IRBUF_WRITE(lut)
IMPLEMENT_GLB_IRBUF_WRITE(matrix_band_part)
IMPLEMENT_GLB_IRBUF_WRITE(conv3d)
IMPLEMENT_GLB_IRBUF_WRITE(pool3d)
IMPLEMENT_GLB_IRBUF_WRITE(stridecalc)
IMPLEMENT_GLB_IRBUF_WRITE(interleave)
IMPLEMENT_GLB_IRBUF_WRITE(bitwise)
IMPLEMENT_GLB_IRBUF_WRITE(binary_shift)
IMPLEMENT_GLB_IRBUF_WRITE(gru)
IMPLEMENT_GLB_IRBUF_WRITE(pytorch_lstm)
IMPLEMENT_GLB_IRBUF_WRITE(tpu)
IMPLEMENT_GLB_IRBUF_WRITE(upsample_mask)
IMPLEMENT_GLB_IRBUF_WRITE(upsample)
IMPLEMENT_GLB_IRBUF_WRITE(group_norm)
IMPLEMENT_GLB_IRBUF_WRITE(deconv3d)
// global_dynamic step 5: implement static_glb_xxx_irbuf_write

// the irbuf will be unpacked by dynamic_glb_arith_shift_layer_ctrl(), these two
// functions should exactly match on the format of this irbuf chunk
void *static_glb_arith_shift_irbuf_write(void *p_ir_buf,
                                         ir_layer_info_t *ir_layer_info) {
#if (0)
  printf("%s(): layer_id=%d, fw_layer_type=%d\n", __func__,
         ir_layer_info->layer_id, ir_layer_info->fw_layer_type);
  for (u32 i = 0; i < ir_layer_info->ir_tensor_info_v.size(); i++) {
    printf(
        "%s(): tensor[%d]: id=%d, addr=%lld, is_io_tensor=%d, tensor_type=%d\n",
        __func__, i, ir_layer_info->ir_tensor_info_v[i].tensor_id,
        ir_layer_info->ir_tensor_info_v[i].global_mem_offset,
        ir_layer_info->ir_tensor_info_v[i].is_io_tensor,
        ir_layer_info->ir_tensor_info_v[i].tensor_type);
  }
  fflush(stdout);
#endif

  void *p_ir_addr = p_ir_buf;
  fw_arith_shift_layer_param_t layer_param =
      ir_layer_info->fw_layer_param_u.fw_arith_shift_layer_param;

  // layer parameter
  *(fw_arith_shift_layer_param_t *)p_ir_addr = layer_param;
  p_ir_addr = (fw_arith_shift_layer_param_t *)p_ir_addr + 1;

  // 1st input (tensor)
  p_ir_addr = write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[0]);

  if (layer_param.shift_is_const == 0) {
    // 2st input
    if (layer_param.is_num_neuron == 1) {
      p_ir_addr =
          write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[1]);
    } else {
      // 2nd input (coeff), only store its offset
      *(u64 *)p_ir_addr = ir_layer_info->ir_tensor_info_v[1].global_mem_offset;
      p_ir_addr = (u64 *)p_ir_addr + 1;
    }

    // the output (tensor)
    p_ir_addr =
        write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[2]);
  } else {

    p_ir_addr =
        write_tensor_info(p_ir_addr, &ir_layer_info->ir_tensor_info_v[1]);
  }

  return p_ir_addr;
}

} // namespace tpu
} // namespace tpu_mlir

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynIrInfo.hpp"

using namespace std;

namespace tpu_mlir {
namespace tpu {

void *call_global_layer_ir_write(FW_LAYER_TYPE_T fw_layer_type, void *p_ir_addr,
                                 ir_layer_info_t *ir_layer_info);

void *static_glb_fc_irbuf_write(void *p_ir_buf, ir_layer_info_t *ir_layer_info);

void *static_glb_batchnorm_irbuf_write(void *p_ir_buf,
                                       ir_layer_info_t *ir_layer_info);

void *static_glb_scale_irbuf_write(void *p_ir_buf,
                                   ir_layer_info_t *ir_layer_info);

void *static_glb_eltwise_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *ir_layer_info);

void *static_glb_softmax_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *ir_layer_info);

void *static_glb_prelu_irbuf_write(void *p_ir_buf,
                                   ir_layer_info_t *ir_layer_info);

void *static_glb_pooling_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *ir_layer_info);

void *static_glb_conv_irbuf_write(void *p_ir_buf,
                                  ir_layer_info_t *ir_layer_info);

void *static_glb_reshape_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *ir_layer_info);

void *static_glb_permute_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *ir_layer_info);

void *static_glb_crop_irbuf_write(void *p_ir_buf,
                                  ir_layer_info_t *ir_layer_info);

void *static_glb_rpn_irbuf_write(void *p_ir_buf,
                                 ir_layer_info_t *ir_layer_info);

void *static_glb_pooling_tf_irbuf_write(void *p_ir_buf,
                                        ir_layer_info_t *ir_layer_info);

void *static_glb_split_tf_irbuf_write(void *p_ir_buf,
                                      ir_layer_info_t *ir_layer_info);

void *static_glb_deconv_irbuf_write(void *p_ir_buf,
                                    ir_layer_info_t *ir_layer_info);

void *static_glb_pad_irbuf_write(void *p_ir_buf,
                                 ir_layer_info_t *ir_layer_info);
void *static_glb_arg_irbuf_write(void *p_ir_buf,
                                 ir_layer_info_t *ir_layer_info);

void *static_glb_reduce_irbuf_write(void *p_ir_buf,
                                    ir_layer_info_t *ir_layer_info);
void *static_glb_select_irbuf_write(void *p_ir_buf,
                                    ir_layer_info_t *ir_layer_info);
void *static_glb_tile_irbuf_write(void *p_ir_buf,
                                  ir_layer_info_t *ir_layer_info);
void *static_glb_broadcast_binary_irbuf_write(void *p_ir_buf,
                                              ir_layer_info_t *ir_layer_info);
void *static_glb_eltwise_binary_irbuf_write(void *p_ir_buf,
                                            ir_layer_info_t *ir_layer_info);
void *static_glb_const_binary_irbuf_write(void *p_ir_buf,
                                          ir_layer_info_t *ir_layer_info);
void *static_glb_biasadd_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *ir_layer_info);

void *static_glb_active_irbuf_write(void *p_ir_buf,
                                    ir_layer_info_t *ir_layer_info);

void *static_glb_normalize_irbuf_write(void *p_ir_buf,
                                       ir_layer_info_t *ir_layer_info);

void *static_glb_priorbox_irbuf_write(void *p_ir_buf,
                                      ir_layer_info_t *ir_layer_info);

void *static_glb_mulshift_irbuf_write(void *p_ir_buf,
                                      ir_layer_info_t *ir_layer_info);

#define DECLARE_GLB_IRWRITE_FUNC(name)                                         \
  void *static_glb_##name##_irbuf_write(void *p_ir_buf,                        \
                                        ir_layer_info_t *ir_layer_info);

DECLARE_GLB_IRWRITE_FUNC(shape_const)
DECLARE_GLB_IRWRITE_FUNC(shape_ref)
DECLARE_GLB_IRWRITE_FUNC(shape_op)
DECLARE_GLB_IRWRITE_FUNC(shape_slice)
DECLARE_GLB_IRWRITE_FUNC(shape_pack)
DECLARE_GLB_IRWRITE_FUNC(shape_assign)
DECLARE_GLB_IRWRITE_FUNC(shape_reorder)
DECLARE_GLB_IRWRITE_FUNC(expand_dims)
DECLARE_GLB_IRWRITE_FUNC(where)
DECLARE_GLB_IRWRITE_FUNC(masked_fill)
DECLARE_GLB_IRWRITE_FUNC(masked_select)
DECLARE_GLB_IRWRITE_FUNC(index_select)
DECLARE_GLB_IRWRITE_FUNC(sort_per_dim)
DECLARE_GLB_IRWRITE_FUNC(nms)
DECLARE_GLB_IRWRITE_FUNC(squeeze)
DECLARE_GLB_IRWRITE_FUNC(ref_pad)
DECLARE_GLB_IRWRITE_FUNC(ref_crop)
DECLARE_GLB_IRWRITE_FUNC(transpose)
DECLARE_GLB_IRWRITE_FUNC(reduce_full)
DECLARE_GLB_IRWRITE_FUNC(concat)
DECLARE_GLB_IRWRITE_FUNC(stride_slice)
DECLARE_GLB_IRWRITE_FUNC(batch2space)
DECLARE_GLB_IRWRITE_FUNC(space2batch)
DECLARE_GLB_IRWRITE_FUNC(output)
DECLARE_GLB_IRWRITE_FUNC(interp)
DECLARE_GLB_IRWRITE_FUNC(expand)
DECLARE_GLB_IRWRITE_FUNC(embedding)
DECLARE_GLB_IRWRITE_FUNC(topk)
DECLARE_GLB_IRWRITE_FUNC(cumsum)
DECLARE_GLB_IRWRITE_FUNC(shape_addn)
DECLARE_GLB_IRWRITE_FUNC(roi_pooling)
DECLARE_GLB_IRWRITE_FUNC(psroi_pooling)
DECLARE_GLB_IRWRITE_FUNC(lrn)
DECLARE_GLB_IRWRITE_FUNC(constant_fill)
DECLARE_GLB_IRWRITE_FUNC(simple_crop)
DECLARE_GLB_IRWRITE_FUNC(slicelike)
DECLARE_GLB_IRWRITE_FUNC(adaptive_pool)
DECLARE_GLB_IRWRITE_FUNC(batch_matmul)
DECLARE_GLB_IRWRITE_FUNC(where_squeeze_gather)

DECLARE_GLB_IRWRITE_FUNC(shape_range)
DECLARE_GLB_IRWRITE_FUNC(range)
DECLARE_GLB_IRWRITE_FUNC(shape_tile)
DECLARE_GLB_IRWRITE_FUNC(shape_reverse)
DECLARE_GLB_IRWRITE_FUNC(shape_expand_ndims)
DECLARE_GLB_IRWRITE_FUNC(shape_cast)
DECLARE_GLB_IRWRITE_FUNC(shape_reshape)
DECLARE_GLB_IRWRITE_FUNC(shape_reduce)
DECLARE_GLB_IRWRITE_FUNC(dtype_convert)
DECLARE_GLB_IRWRITE_FUNC(yolo)
DECLARE_GLB_IRWRITE_FUNC(yolov3_detect_out)
DECLARE_GLB_IRWRITE_FUNC(ssd_detect_out)
DECLARE_GLB_IRWRITE_FUNC(host2device)
DECLARE_GLB_IRWRITE_FUNC(device2host)
DECLARE_GLB_IRWRITE_FUNC(tensorarray)
DECLARE_GLB_IRWRITE_FUNC(ta_size)
DECLARE_GLB_IRWRITE_FUNC(ta_read)
DECLARE_GLB_IRWRITE_FUNC(ta_write)
DECLARE_GLB_IRWRITE_FUNC(ta_scatter)
DECLARE_GLB_IRWRITE_FUNC(ta_gather)
DECLARE_GLB_IRWRITE_FUNC(ta_split)
DECLARE_GLB_IRWRITE_FUNC(ta_concat)
DECLARE_GLB_IRWRITE_FUNC(shape_split)
DECLARE_GLB_IRWRITE_FUNC(shape_unary)
DECLARE_GLB_IRWRITE_FUNC(shape_squeeze)
DECLARE_GLB_IRWRITE_FUNC(rank)
DECLARE_GLB_IRWRITE_FUNC(slice)
DECLARE_GLB_IRWRITE_FUNC(shape_sizeslice)
DECLARE_GLB_IRWRITE_FUNC(coeff2neuron)
DECLARE_GLB_IRWRITE_FUNC(shape_select)
DECLARE_GLB_IRWRITE_FUNC(depth2space)
DECLARE_GLB_IRWRITE_FUNC(reverse)
DECLARE_GLB_IRWRITE_FUNC(lstm)
DECLARE_GLB_IRWRITE_FUNC(broadcast_like)
DECLARE_GLB_IRWRITE_FUNC(reorg)
DECLARE_GLB_IRWRITE_FUNC(lut)
DECLARE_GLB_IRWRITE_FUNC(matrix_band_part)
DECLARE_GLB_IRWRITE_FUNC(conv3d)
DECLARE_GLB_IRWRITE_FUNC(pool3d)
DECLARE_GLB_IRWRITE_FUNC(stridecalc)
DECLARE_GLB_IRWRITE_FUNC(interleave)
DECLARE_GLB_IRWRITE_FUNC(bitwise)
DECLARE_GLB_IRWRITE_FUNC(binary_shift)
DECLARE_GLB_IRWRITE_FUNC(gru)
DECLARE_GLB_IRWRITE_FUNC(pytorch_lstm)
DECLARE_GLB_IRWRITE_FUNC(arith_shift)
DECLARE_GLB_IRWRITE_FUNC(multi_masked_select)
DECLARE_GLB_IRWRITE_FUNC(tpu)
DECLARE_GLB_IRWRITE_FUNC(sequence_gen)
DECLARE_GLB_IRWRITE_FUNC(upsample_mask)
DECLARE_GLB_IRWRITE_FUNC(upsample)
DECLARE_GLB_IRWRITE_FUNC(group_norm)
DECLARE_GLB_IRWRITE_FUNC(deconv3d)
// global_dynamic : declare static_glb_xxx_irbuf_write here

#undef DECLARE_GLB_IRWRITE_FUNC

} // namespace tpu
} // namespace tpu_mlir

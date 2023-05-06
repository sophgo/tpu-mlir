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

void *call_local_layer_ir_write(FW_LAYER_TYPE_T fw_layer_type, void *p_ir_addr,
                                ir_layer_info_t *p_ir_layer_info);

void *static_loc_mulshift_irbuf_write(void *p_ir_buf,
                                      ir_layer_info_t *p_ir_layer_info);

void *static_loc_conv_irbuf_write(void *p_ir_buf,
                                  ir_layer_info_t *p_ir_layer_info);

void *static_loc_pooling_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *p_ir_layer_info);

void *static_loc_lrn_irbuf_write(void *p_ir_buf,
                                 ir_layer_info_t *p_ir_layer_info);

void *static_loc_prelu_irbuf_write(void *p_ir_buf,
                                   ir_layer_info_t *p_ir_layer_info);

void *static_loc_eltwise_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *p_ir_layer_info);

void *static_loc_scale_irbuf_write(void *p_ir_buf,
                                   ir_layer_info_t *p_ir_layer_info);

void *static_loc_batchnorm_irbuf_write(void *p_ir_buf,
                                       ir_layer_info_t *p_ir_layer_info);

void *static_loc_deconv_irbuf_write(void *p_ir_buf,
                                    ir_layer_info_t *p_ir_layer_info);

void *static_loc_concat_irbuf_write(void *p_ir_buf,
                                    ir_layer_info_t *p_ir_layer_info);

void *static_loc_pooling_tf_irbuf_write(void *p_ir_buf,
                                        ir_layer_info_t *p_ir_layer_info);

void *static_loc_active_irbuf_write(void *p_ir_buf,
                                    ir_layer_info_t *p_ir_layer_info);

void *static_loc_crop_irbuf_write(void *p_ir_buf,
                                  ir_layer_info_t *p_ir_layer_info);

void *static_loc_broadcast_binary_irbuf_write(void *p_ir_buf,
                                              ir_layer_info_t *p_ir_layer_info);
void *static_loc_const_binary_irbuf_write(void *p_ir_buf,
                                          ir_layer_info_t *p_ir_layer_info);
void *static_loc_eltwise_binary_irbuf_write(void *p_ir_buf,
                                            ir_layer_info_t *p_ir_layer_info);
void *static_loc_biasadd_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *p_ir_layer_info);
void *static_loc_select_irbuf_write(void *p_ir_buf,
                                    ir_layer_info_t *p_ir_layer_info);
void *static_loc_upsample_irbuf_write(void *p_ir_buf,
                                      ir_layer_info_t *p_ir_layer_info);
void *static_loc_strideslice_irbuf_write(void *p_ir_buf,
                                         ir_layer_info_t *p_ir_layer_info);
void *static_loc_reorg_irbuf_write(void *p_ir_buf,
                                   ir_layer_info_t *p_ir_layer_info);
void *static_loc_dtype_convert_irbuf_write(void *p_ir_buf,
                                           ir_layer_info_t *p_ir_layer_info);
void *static_loc_tile_irbuf_write(void *p_ir_buf,
                                  ir_layer_info_t *p_ir_layer_info);
void *static_loc_pad_irbuf_write(void *p_ir_buf,
                                 ir_layer_info_t *p_ir_layer_info);
void *static_loc_reduce_full_irbuf_write(void *p_ir_buf,
                                         ir_layer_info_t *p_ir_layer_info);
void *static_loc_lut_irbuf_write(void *p_ir_buf,
                                 ir_layer_info_t *p_ir_layer_info);
void *static_loc_stridecalc_irbuf_write(void *p_ir_buf,
                                        ir_layer_info_t *p_ir_layer_info);
void *static_loc_interleave_irbuf_write(void *p_ir_buf,
                                        ir_layer_info_t *p_ir_layer_info);
void *static_loc_bitwise_irbuf_write(void *p_ir_buf,
                                     ir_layer_info_t *p_ir_layer_info);
void *static_loc_arith_shift_irbuf_write(void *p_ir_buf,
                                         ir_layer_info_t *p_ir_layer_info);
void *static_loc_normalize_irbuf_write(void *p_ir_buf,
                                       ir_layer_info_t *p_ir_layer_info);
void *static_loc_binary_shift_irbuf_write(void *p_ir_buf,
                                          ir_layer_info_t *p_ir_layer_info);
void *static_loc_arg_irbuf_write(void *p_ir_buf,
                                 ir_layer_info_t *p_ir_layer_info);
void *static_loc_tpu_irbuf_write(void *p_ir_buf,
                                 ir_layer_info_t *p_ir_layer_info);
void *static_loc_conv3d_irbuf_write(void *p_ir_buf,
                                    ir_layer_info_t *p_ir_layer_info);
void *static_loc_pool3d_irbuf_write(void *p_ir_buf,
                                    ir_layer_info_t *p_ir_layer_info);
void *static_loc_group_norm_irbuf_write(void *p_ir_buf,
                                        ir_layer_info_t *p_ir_layer_info);
// local_dynamic : declare irbuf write function here

} // namespace tpu
} // namespace tpu_mlir

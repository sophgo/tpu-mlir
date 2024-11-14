//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::CastOp::codegen_global_bm1684() {
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  auto input_dtype = BM1684::getDataType(getInput());
  auto output_dtype = BM1684::getDataType(getOutput());
  if (output_dtype == DTYPE_FP32 &&
      (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8)) {
    // int8 => fp32
    BM1684::instance().dl_nodechip_global_int2float(
        module::getAddress(getInput()), module::getAddress(getOutput()), n, c,
        h, w, input_dtype == DTYPE_INT8 ? 1 : 0, 1.f, STORAGE_MODE_4N_INT8,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else if (input_dtype == DTYPE_FP32 &&
             (output_dtype == DTYPE_INT8 || output_dtype == DTYPE_UINT8)) {
    // fp32 => int8
    BM1684::instance().dl_nodechip_float2int8_v2(
        module::getAddress(getInput()), module::getAddress(getOutput()), n, c,
        h, w, output_dtype == DTYPE_INT8 ? 1 : 0, 1.f, STORAGE_MODE_4N_INT8,
        ROUND_INF, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else if (input_dtype == DTYPE_INT32 && output_dtype == DTYPE_FP32) {
    // int32 => fp32
    BM1684::instance().dl_nodechip_unary(
        module::getAddress(getInput()), module::getAddress(getOutput()),
        module::getNumElements(getInput()), UNARY_I32_TO_F32, NULL,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else if (input_dtype == DTYPE_FP32 && output_dtype == DTYPE_INT32) {
    // fp32 => int32
    uint32_t input_shape[MAX_SHAPE_DIMS];
    module::getGlobalShape(getInput(), (int *)input_shape);
    BM1684::instance().dl_nodechip_float_to_int32_global(
        module::getAddress(getInput()), module::getAddress(getOutput()),
        input_shape, module::getShape(getInput()).size(), 1 /*input sign*/,
        1 /*output sign*/, STORAGE_MODE_1N_FP32, STORAGE_MODE_1N_FP32,
        ROUND_INF, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    dump();
    llvm_unreachable("CastOp type error");
  }
}

int64_t tpu::CastOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  int64_t local_buffer_size = 0;
  auto input_dtype = BM1684::getDataType(getInput());
  auto output_dtype = BM1684::getDataType(getOutput());
  if (input_dtype == DTYPE_FP32) {
    local_buffer_size = in_lmem_bytes * 2;
  } else if (output_dtype == DTYPE_FP32 &&
             (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8)) {
    int64_t n, c, h, w;
    module::getNCHW(getInput(), n, c, h, w);
    int dsize = sizeof(float);
    local_buffer_size = ceiling_func(in_nslice, (int64_t)4) *
                        ceiling_func(c, BM1684::NPU_NUM) *
                        align_up(in_hslice * w, BM1684::eu_num(dsize)) * dsize;
  } else {
    llvm_unreachable("CastOp buffer type error");
  }
  return local_buffer_size;
}

void tpu::CastOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                       local_sec_info_t &sec_info) {
  auto input_dtype = BM1684::getDataType(getInput());
  auto output_dtype = BM1684::getDataType(getOutput());
  auto input_group_info =
      LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto output_group_info = getGroupInfo(n_step, h_step, 0, 0, 0);
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  uint32_t input_shape[4] = {(uint32_t)input_group_info.n_slice, (uint32_t)c,
                             (uint32_t)input_group_info.h_slice, (uint32_t)w};
  if (output_dtype == DTYPE_FP32 &&
      (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8)) {
    // fix8b => fp32
    BM1684::instance().dl_tensor_int8_to_float_local_v2(
        input_group_info.out_addr, output_group_info.out_addr,
        output_group_info.buffer_addr, input_shape[0], input_shape[1],
        input_shape[2], input_shape[3], true, input_dtype == DTYPE_INT8 ? 1 : 0,
        STORAGE_MODE_4N_INT8, BM1684::instance()->bdc_node);
  } else if (input_dtype == DTYPE_FP32 &&
             (output_dtype == DTYPE_INT8 || output_dtype == DTYPE_UINT8)) {
    // fp32 => fix8b
    BM1684::instance().dl_nodechip_float2int8_local_keep_input(
        input_group_info.out_addr, output_group_info.out_addr,
        output_group_info.buffer_addr, input_shape[0], input_shape[1],
        input_shape[2], input_shape[3], output_dtype == DTYPE_INT8 ? 1 : 0, 0,
        ROUND_INF, BM1684::instance()->bdc_node);
  } else if (input_dtype == DTYPE_FP32 && output_dtype == DTYPE_INT32) {
    // fp32 => int32
    BM1684::instance().dl_nodechip_float_to_int32_local(
        input_group_info.out_addr, output_group_info.out_addr,
        output_group_info.buffer_addr, input_shape, 1 /*input sign*/,
        1 /*output sign*/, STORAGE_MODE_1N_FP32, STORAGE_MODE_1N_FP32,
        ROUND_INF, (CMD_ID_NODE *)BM1684::instance()->bdc_node);
  } else {
    llvm_unreachable("CastOp type error");
  }
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================

uint32_t tpu::CastOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  uint32_t fw_ir_length = 0;
  ir_layer_info_t *cast_layer_info = (ir_layer_info_t *)ir_layer_info;
  dynamic_common_ir_layer_info(cast_layer_info, getInput(), getOutput());
  cast_layer_info->data_size =
      get_dynamic_compiler_tensor_datasize(getOutput());
  assign_fw_param(
      (void *)&cast_layer_info->fw_layer_param_u.fw_dtype_convert_layer_param);
  fw_ir_length += sizeof(fw_dtype_convert_layer_param_t);
  return fw_ir_length;
}

int64_t tpu::CastOp::get_fw_type_bm1684() { return FW_BMNET_DTYPE_CONVERT; }

// ======================================
// Dynamic LocalGenInterface
// ======================================

int32_t tpu::CastOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  int fw_ir_length = 0;
  ir_layer_info_t *cast_layer_info = (ir_layer_info_t *)ir_layer_info;
  dynamic_common_ir_layer_info(cast_layer_info, getInput(), getOutput());
  cast_layer_info->data_size =
      get_dynamic_compiler_tensor_datasize(getOutput());
  assign_fw_param(
      (void *)&cast_layer_info->fw_layer_param_u.fw_dtype_convert_layer_param);
  fw_ir_length += sizeof(fw_dtype_convert_layer_param_t);

  // input tensor
  dynamic_push_back_local_tensor(cast_layer_info->ir_tensor_info_v, getInput());

  // output
  dynamic_push_back_local_tensor(cast_layer_info->ir_tensor_info_v,
                                 getOutput());

  // imm buffer
  dynamic_push_back_local_buffer(cast_layer_info->ir_tensor_info_v, 0,
                                 getOutput());

  // compute fw ir info length for input and output
  fw_ir_length +=
      2 * sizeof(uint32_t) + 2 * sizeof(uint32_t) + sizeof(uint32_t);

  // add fw ir length for output consumer number
  fw_ir_length += sizeof(uint32_t);

  return fw_ir_length;
}

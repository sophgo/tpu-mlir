//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::MaxConstOp::codegen_global_bm1684() {
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  if (module::isUniformQuantized(getInput())) {
    int bottom_shapes[MAX_SHAPE_DIMS] = {(int)n, (int)c, (int)h, (int)w};
    int is_int8[3] = {1, 0, 1};
    int is_sign[3] = {module::isSign(getInput()), 1,
                      module::isSign(getOutput())};
    BM1684::instance().dl_nodechip_const_binary_fix8b_forward_parallel(
        module::getAddress(getInput()), module::getAddress(getOutput()),
        getConstVal().convertToDouble(), bottom_shapes,
        module::getShape(getInput()).size(), BINARY_MAX, getMultiplier(), 1,
        getRshift(), 0,
        /*inversed*/ 0, getDoRelu(), is_int8, is_sign,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    BM1684::instance().dl_nodechip_const_binary(
        module::getAddress(getInput()), n * c * h * w,
        getConstVal().convertToDouble(), module::getAddress(getOutput()),
        BINARY_MAX, 0, getDoRelu(), getReluLimit().convertToDouble(),
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node,
        module::getStorageType(getInput()).isa<IntegerType>());
  }
}

int64_t tpu::MaxConstOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::MaxConstOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                           local_sec_info_t &sec_info) {
  auto out_g_info = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto in_g_info = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  int b0_shape[MAX_SHAPE_DIMS] = {(int)in_g_info.n_slice, (int)c,
                                  (int)in_g_info.h_slice, (int)w};
  auto b1_val = (float)getConstVal().convertToDouble();
  auto if_relu = getDoRelu();
  auto relu_limit = getReluLimit().convertToDouble();
  if (module::isUniformQuantized(getOutput())) {
    int is_signs[3] = {module::isSign(getInput()), 1,
                       module::isSign(getOutput())};
    int is_int8s[3] = {1, 0, 1};
    BM1684::instance().dl_nodechip_const_binary_fix8b_forward_local(
        in_g_info.out_addr, out_g_info.out_addr, /*imm_buffer*/ 0,
        getConstVal().convertToDouble(), b0_shape,
        module::getShape(getInput()).size(), BINARY_MAX, getMultiplier(), 1,
        getRshift(), 0, /*inversed*/ 0, getDoRelu(), is_int8s, is_signs,
        BM1684::instance()->bdc_node);
  } else {
    BM1684::instance().dl_nodechip_const_binary_local(
        in_g_info.out_addr, (uint32_t *)b0_shape, b1_val, out_g_info.out_addr,
        BINARY_MAX, 0, if_relu, relu_limit,
        (CMD_ID_NODE *)BM1684::instance()->bdc_node);
  }
}

uint32_t tpu::MaxConstOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  GLOBAL_IR_COMMON(const_binary);
}

int64_t tpu::MaxConstOp::get_fw_type_bm1684() { return FW_BMNET_CONST_BINARY; }

int32_t tpu::MaxConstOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  int fw_ir_length = 0;
  IR_PARAM_COMMON(const_binary);
  // input tensor
  dynamic_push_back_local_tensor(layer_info->ir_tensor_info_v, getInput());
  // output tensor
  dynamic_push_back_local_tensor(layer_info->ir_tensor_info_v, getOutput());
  if (DSIZE_FP32 != layer_info->data_size) {
    dynamic_push_back_local_buffer(layer_info->ir_tensor_info_v,
                                   get_tensor_id(getInput()), getOutput());
    fw_ir_length += sizeof(uint32_t);
  }
  // compute fw ir info length for input and output
  fw_ir_length += 2 * 2 * sizeof(uint32_t);
  // add fw ir length for output consumer number
  fw_ir_length += sizeof(uint32_t);
  return fw_ir_length;
}

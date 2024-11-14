//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

using namespace tpu_mlir::backend;

static int isInt8(Value v) {
  auto dType = BM168x::getDataType(v);
  return dType == DTYPE_INT8 || dType == DTYPE_UINT8;
}

void tpu::MaxOp::codegen_global_bm1684() {
  int input_num = getInputs().size();
  assert(input_num == 2);
  int op_code = BINARY_MAX;
  auto b0_addr = module::getAddress(getInputs()[0]);
  auto b1_addr = module::getAddress(getInputs()[1]);
  auto top_addr = module::getAddress(getOutput());
  int b0_shape[MAX_SHAPE_DIMS];
  int b1_shape[MAX_SHAPE_DIMS];
  module::getGlobalShape(getInputs()[0], b0_shape);
  module::getGlobalShape(getInputs()[1], b1_shape);
  auto b0_dims = module::getShape(getInputs()[0]).size();
  auto b1_dims = module::getShape(getInputs()[1]).size();
  if (false == module::isUniformQuantized(getOutput())) {
    BM1684::instance().dl_nodechip_broadcast_binary_full(
        b0_addr, (uint32_t *)b0_shape, b0_dims, b1_addr, (uint32_t *)b1_shape,
        b1_dims, top_addr, 0, op_code, getDoRelu(), -1.f, 0,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node, DTYPE_FP32);
  } else {
    int is_int8[3] = {isInt8(getInputs()[0]), isInt8(getInputs()[1]),
                      isInt8(getOutput())};
    int is_sign[3] = {module::isSign(getInputs()[0]),
                      module::isSign(getInputs()[1]),
                      module::isSign(getOutput())};
    auto muls = module::getI64Array(getMultipliers(), input_num, 1);
    auto rs = module::getI64Array(getRshifts(), input_num, 0);
    BM1684::instance().dl_nodechip_broadcast_binary_fix8b_forward_parallel(
        b0_addr, b1_addr, top_addr, b0_shape, b1_shape,
        module::getShape(getInputs()[0]).size(),
        module::isWeight(getInputs()[0]), module::isWeight(getInputs()[1]),
        op_code, muls->at(0), muls->at(1), rs->at(0), rs->at(1), is_int8,
        is_sign, getDoRelu(), (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

int64_t tpu::MaxOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes,
                                         int64_t in_nslice, int64_t in_hslice,
                                         int64_t out_nslice,
                                         int64_t out_hslice) {
  return 0;
}

void tpu::MaxOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                      local_sec_info_t &sec_info) {
  int op_code = BINARY_MAX;
  int input_num = getInputs().size();
  int64_t t_n, t_c, t_h, t_w;
  int64_t b0_n, b0_c, b0_h, b0_w;
  int64_t b1_n, b1_c, b1_h, b1_w;
  module::getNCHW(getOutput(), t_n, t_c, t_h, t_w);
  module::getNCHW(getInputs()[0], b0_n, b0_c, b0_h, b0_w);
  module::getNCHW(getInputs()[1], b1_n, b1_c, b1_h, b1_w);
  auto top_ginfo = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto b0_ginfo = LocalGenInterface::getGroupInfo(getInputs()[0]);
  auto b1_ginfo = LocalGenInterface::getGroupInfo(getInputs()[1]);
  int b0_shape[MAX_SHAPE_DIMS] = {(int)b0_ginfo.n_slice, (int)b0_c,
                                  (int)b0_ginfo.h_slice, (int)b0_w};
  int b1_shape[MAX_SHAPE_DIMS] = {(int)b1_ginfo.n_slice, (int)b1_c,
                                  (int)b1_ginfo.h_slice, (int)b1_w};
  auto b0_dims = module::getShape(getInputs()[0]).size();
  if (module::isUniformQuantized(getOutput())) {
    int is_sign[3] = {isInt8(getInputs()[0]), isInt8(getInputs()[1]),
                      isInt8(getOutput())};
    int is_int8[3] = {module::isSign(getInputs()[0]),
                      module::isSign(getInputs()[1]),
                      module::isSign(getOutput())};
    auto muls = module::getI64Array(getMultipliers(), input_num, 1);
    auto rs = module::getI64Array(getRshifts(), input_num, 0);
    BM1684::instance().dl_nodechip_broadcast_binary_fix8b_forward_local(
        b0_ginfo.out_addr, b1_ginfo.out_addr, top_ginfo.out_addr,
        /*top buffer_addr*/ 0, b0_shape, b1_shape, b0_dims,
        module::isWeight(getInputs()[0]), module::isWeight(getInputs()[1]),
        op_code, muls->at(0), muls->at(1), rs->at(0), rs->at(1), is_int8,
        is_sign, getDoRelu(), BM1684::instance()->bdc_node);
  } else {
    int b0_stride[4];
    int b1_stride[4];
    int top_stride[4];
    int top_shape[MAX_SHAPE_DIMS] = {(int)top_ginfo.n_slice, (int)t_c,
                                     (int)top_ginfo.h_slice, (int)t_w};
    module::get128BtyeAlignedStrideForNBit(b0_stride, b0_shape, BM1684::NPU_NUM,
                                           32);
    module::get128BtyeAlignedStrideForNBit(b1_stride, b1_shape, BM1684::NPU_NUM,
                                           32);
    module::get128BtyeAlignedStrideForNBit(top_stride, top_shape,
                                           BM1684::NPU_NUM, 32);
    for (int i = 0; i < 4; i++) {
      if (b0_shape[i] == 1 && b1_shape[i] != 1)
        b0_stride[i] = 0;
      else if (b0_shape[i] != 1 && b1_shape[i] == 1)
        b1_stride[i] = 0;
    }
    BM1684::instance().dl_nodechip_broadcast_binary_local(
        b0_ginfo.out_addr, b0_shape, b0_stride, b1_ginfo.out_addr, b1_shape,
        b1_stride, top_ginfo.out_addr, top_stride, op_code, getDoRelu(),
        (float)getReluLimit().convertToDouble(),
        b0_shape[1] > b1_shape[1] ? b1_ginfo.out_addr : b0_ginfo.out_addr,
        BM1684::instance()->bdc_node);
  }
}

uint32_t tpu::MaxOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  ir_layer_info_t *add_layer_info = (ir_layer_info_t *)ir_layer_info;
  fw_broadcast_binary_layer_param_t fw_broadcast_binary_layer_param = {0};
  dynamic_common_ir_layer_info(add_layer_info, getInputs()[0], getOutput());
  assign_fw_param((void *)&fw_broadcast_binary_layer_param);
  add_layer_info->fw_layer_param_u.fw_broadcast_binary_layer_param =
      fw_broadcast_binary_layer_param;
  return sizeof(fw_broadcast_binary_layer_param_t);
}
int64_t tpu::MaxOp::get_fw_type_bm1684() { return FW_BMNET_BROADCAST_BINARY; }

int32_t tpu::MaxOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

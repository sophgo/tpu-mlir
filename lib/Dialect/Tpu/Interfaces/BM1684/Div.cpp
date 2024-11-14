//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::DivOp::codegen_global_bm1684() {
  int input_num = getInputs().size();
  assert(input_num == 2);
  int op_code = 3;
  auto a_addr = module::getAddress(getInputs()[0]);
  auto b_addr = module::getAddress(getInputs()[1]);
  auto o_addr = module::getAddress(getOutput());
  int a_shape[MAX_SHAPE_DIMS] = {1};
  int b_shape[MAX_SHAPE_DIMS] = {1};
  auto a_dims = module::getShape(getInputs()[0]).size();
  auto b_dims = module::getShape(getInputs()[1]).size();
  module::getGlobalShape(getInputs()[0], a_shape);
  module::getGlobalShape(getInputs()[1], b_shape);
  if (false == module::isUniformQuantized(getOutput())) {
    auto dtype = BM1684::getDataType(getOutput());
    int src_int32 = dtype == DTYPE_FP32 ? 0 : 1;
    auto gdma_format = BM1684::GDMA_VALUE_FORMAT_FLOAT32;
    auto buffer_size = BM1684::instance().dl_get_broadcast_binary_buffer_size(
        (uint32_t *)a_shape, a_dims, (uint32_t *)b_shape, b_dims,
        sizeof(float));
    if (buffer_size) {
      llvm_unreachable("Need Create Global Buffer");
    }
    BM1684::instance().dl_nodechip_broadcast_binary_full(
        a_addr, (uint32_t *)a_shape, a_dims, b_addr, (uint32_t *)b_shape,
        b_dims, o_addr, 0 /*buffer_addr, special case may use*/, op_code,
        getDoRelu(), getReluLimit().convertToDouble(), gdma_format,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node, src_int32);
  }
}

int64_t tpu::DivOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes,
                                         int64_t in_nslice, int64_t in_hslice,
                                         int64_t out_nslice,
                                         int64_t out_hslice) {
  int64_t buffer_size = 0;
  return buffer_size;
}

void tpu::DivOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                      local_sec_info_t &sec_info) {
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  auto out_gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  int num_inputs = getInputs().size();
  llvm::SmallVector<int, 8> input_addrs;
  int op_code = 3;
  for (int i = 0; i < num_inputs; i++) {
    auto in = getInputs()[i];
    auto in_ginfo = LocalGenInterface::getGroupInfo(in);
    input_addrs.push_back(in_ginfo.out_addr);
  }
  auto in0_g_info =
      LocalGenInterface::getGroupInfo(getInputs()[0], n_step, h_step);
  int64_t n0, c0, h0, w0;
  module::getNCHW(getInputs()[0], n0, c0, h0, w0);
  int b0_shape[MAX_SHAPE_DIMS] = {(int)in0_g_info.n_slice, (int)c0,
                                  (int)in0_g_info.h_slice, (int)w0};
  auto in1_g_info =
      LocalGenInterface::getGroupInfo(getInputs()[1], n_step, h_step);
  int64_t n1, c1, h1, w1;
  module::getNCHW(getInputs()[1], n1, c1, h1, w1);
  int b1_shape[MAX_SHAPE_DIMS] = {(int)in1_g_info.n_slice, (int)c1,
                                  (int)in1_g_info.h_slice, (int)w1};
  if (false == module::isUniformQuantized(getOutput())) {
    int b0_stride[4] = {0};
    int b1_stride[4] = {0};
    int top_stride[4] = {0};
    int top_shape[MAX_SHAPE_DIMS] = {0};
    module::getLocalShape(getOutput(), n_step, h_step, top_shape);
    module::get128BtyeAlignedStrideForNBit(b0_stride, b0_shape, BM1684::NPU_NUM,
                                           32);
    module::get128BtyeAlignedStrideForNBit(b1_stride, b1_shape, BM1684::NPU_NUM,
                                           32);
    module::get128BtyeAlignedStrideForNBit(top_stride, top_shape,
                                           BM1684::NPU_NUM, 32);
    for (int i = 0; i < 4; i++) {
      if (b0_shape[i] != b1_shape[i]) {
        if (b0_shape[i] == 1)
          b0_stride[i] = 0;
        if (b1_shape[i] == 1)
          b1_stride[i] = 0;
      }
    }
    BM1684::instance().dl_nodechip_broadcast_binary_local(
        input_addrs[0], b0_shape, b0_stride, input_addrs[1], b1_shape,
        b1_stride, out_gi.out_addr, top_stride, op_code, getDoRelu(),
        getReluLimit().convertToDouble(),
        b0_shape[1] > b1_shape[1] ? input_addrs[1] : input_addrs[0],
        BM1684::instance()->bdc_node);
  }
}

uint32_t tpu::DivOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  ir_layer_info_t *add_layer_info = (ir_layer_info_t *)ir_layer_info;
  fw_broadcast_binary_layer_param_t fw_broadcast_binary_layer_param = {0};
  dynamic_common_ir_layer_info(add_layer_info, getInputs()[0], getOutput());
  assign_fw_param((void *)&fw_broadcast_binary_layer_param);
  add_layer_info->fw_layer_param_u.fw_broadcast_binary_layer_param =
      fw_broadcast_binary_layer_param;
  return sizeof(fw_broadcast_binary_layer_param_t);
}

int32_t tpu::DivOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::DivOp::get_fw_type_bm1684() { return FW_BMNET_BROADCAST_BINARY; }

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

void tpu::AddOp::codegen_global_bm1684() {
  int input_num = getInputs().size();
  assert(input_num == 2);
  auto a_dims = module::getShape(getInputs()[0]).size();
  auto b_dims = module::getShape(getInputs()[1]).size();
  auto out_dims = module::getShape(getOutput()).size();
  auto a_addr = module::getAddress(getInputs()[0]);
  auto b_addr = module::getAddress(getInputs()[1]);
  auto o_addr = module::getAddress(getOutput());
  int a_shape[MAX_SHAPE_DIMS], b_shape[MAX_SHAPE_DIMS];
  module::getGlobalShape(getInputs()[0], a_shape);
  module::getGlobalShape(getInputs()[1], b_shape);
  if (!module::isUniformQuantized(getOutput())) {
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
        b_dims, o_addr, 0 /*buffer_addr, special case may use*/, BINARY_ADD,
        getDoRelu(), getReluLimit().convertToDouble(), gdma_format,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node, src_int32);
  } else {
    SmallVector<int> is_sign(input_num + 1, 0);
    SmallVector<int> is_int8(input_num + 1, 0);

    for (int i = 0; i < input_num; ++i) {
      is_int8[i] = module::getDtypeSize(getInputs()[i]) == 1;
      is_sign[i] = module::isSign(getInputs()[i]);
    }
    is_int8[input_num] = module::getDtypeSize(getOutput()) == 1;
    is_sign[input_num] = module::isSign(getOutput());
    auto muls = module::getI32Array(getMultipliers(), input_num, 1);
    auto rs = module::getI32Array(getRshifts(), input_num, 0);
    /// storage mode: it converts 1N to 4N in backend if input is coeff
    /// so if storage of the input is 4N, just set the flag of coeff is false
    bool is_coeff[2] = {0};
    for (int i = 0; i < getNumOperands(); i++) {
      if (auto castOp =
              dyn_cast_or_null<top::WeightOp>(getInputs()[i].getDefiningOp())) {
        is_coeff[i] =
            castOp.getStoreMode().has_value() && castOp.getStoreMode() != "4N";
      }
    }
    BM1684::instance().dl_nodechip_broadcast_binary_fix8b_forward_parallel(
        a_addr, b_addr, o_addr, a_shape, b_shape, out_dims, is_coeff[0],
        is_coeff[1], BINARY_ADD, muls->at(0), muls->at(1), rs->at(0), rs->at(1),
        is_int8.data(), is_sign.data(), getDoRelu(),
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

int64_t tpu::AddOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes,
                                         int64_t in_nslice, int64_t in_hslice,
                                         int64_t out_nslice,
                                         int64_t out_hslice) {
  auto b0_dsize = module::getDtypeSize(getInputs()[0]);
  auto b1_dsize = module::getDtypeSize(getInputs()[1]);
  auto top_dsize = module::getDtypeSize(getOutput());
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  int64_t buffer_size = 0;
  int64_t tensor_size =
      ceiling_func(out_nslice, (int64_t)2) * ceiling_func(c, BM1684::NPU_NUM) *
      align_up(out_hslice * w, BM1684::eu_num(sizeof(float))) * sizeof(float);
  if (b0_dsize == 1 && b1_dsize == 1) {
    buffer_size += tensor_size * 2;
  } else if (b0_dsize + b1_dsize == 1) {
    llvm_unreachable("AddOp buffer dtype error");
  }
  if (top_dsize == 1) {
    buffer_size += tensor_size;
  }
  return buffer_size;
}

void tpu::AddOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                      local_sec_info_t &sec_info) {
  int num_inputs = getInputs().size();
  assert(num_inputs == 2);
  int shape_dim = 4;
  auto out_gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto in0_g_info =
      LocalGenInterface::getGroupInfo(getInputs()[0], n_step, h_step);
  auto in1_g_info =
      LocalGenInterface::getGroupInfo(getInputs()[1], n_step, h_step);
  int b0_shape[shape_dim], b1_shape[shape_dim];
  module::getLocalShape(getInputs()[0], n_step, h_step, b0_shape);
  module::getLocalShape(getInputs()[1], n_step, h_step, b1_shape);
  if (module::isUniformQuantized(getOutput())) {
    auto muls = module::getI32Array(getMultipliers(), num_inputs, 1);
    auto rs = module::getI32Array(getRshifts(), num_inputs, 0);
    int is_sign[num_inputs + 1];
    memset(is_sign, 0, sizeof(int) * (num_inputs + 1));
    int is_int8[num_inputs + 1];
    memset(is_int8, 0, sizeof(int) * (num_inputs + 1));
    for (int i = 0; i < num_inputs; ++i) {
      is_int8[i] = module::getDtypeSize(getInputs()[i]) == 1;
      is_sign[i] = module::isSign(getInputs()[i]);
    }
    is_int8[num_inputs] = module::getDtypeSize(getOutput()) == 1;
    is_sign[num_inputs] = module::isSign(getOutput());
    BM1684::instance().dl_nodechip_broadcast_binary_fix8b_forward_local(
        in0_g_info.out_addr, in1_g_info.out_addr, out_gi.out_addr,
        out_gi.buffer_addr, b0_shape, b1_shape, shape_dim,
        module::isWeight(getInputs()[0]), module::isWeight(getInputs()[1]),
        BINARY_ADD, muls->at(0), muls->at(1), rs->at(0), rs->at(1), is_int8,
        is_sign, getDoRelu(), BM1684::instance()->bdc_node);
  } else {
    int b0_stride[shape_dim], b1_stride[shape_dim], top_stride[shape_dim],
        top_shape[shape_dim];
    module::getLocalShape(getOutput(), n_step, h_step, top_shape);
    module::get128BtyeAlignedStrideForNBit(b0_stride, b0_shape, BM1684::NPU_NUM,
                                           32);
    module::get128BtyeAlignedStrideForNBit(b1_stride, b1_shape, BM1684::NPU_NUM,
                                           32);
    module::get128BtyeAlignedStrideForNBit(top_stride, top_shape,
                                           BM1684::NPU_NUM, 32);
    bool is_eltwise = true;
    for (int i = 0; i < shape_dim; i++) {
      if (b0_shape[i] != b1_shape[i]) {
        is_eltwise = false;
        if (b0_shape[i] == 1)
          b0_stride[i] = 0;
        if (b1_shape[i] == 1)
          b1_stride[i] = 0;
      }
    }
    if (is_eltwise) {
      BM1684::instance().dl_nodechip_eltwise_binary_local(
          in0_g_info.out_addr, in1_g_info.out_addr, out_gi.out_addr,
          module::isWeight(getInputs()[0]), module::isWeight(getInputs()[1]),
          b0_shape, BINARY_ADD, getDoRelu(), getReluLimit().convertToDouble(),
          BM1684::instance()->bdc_node);
    } else {
      BM1684::instance().dl_nodechip_broadcast_binary_local(
          in0_g_info.out_addr, b0_shape, b0_stride, in1_g_info.out_addr,
          b1_shape, b1_stride, out_gi.out_addr, top_stride, BINARY_ADD,
          getDoRelu(), getReluLimit().convertToDouble(),
          b0_shape[1] > b1_shape[1] ? in1_g_info.out_addr : in0_g_info.out_addr,
          BM1684::instance()->bdc_node);
    }
  }
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================

uint32_t tpu::AddOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  int fw_ir_length = 0;
  ir_layer_info_t *add_layer_info = (ir_layer_info_t *)ir_layer_info;
  fw_broadcast_binary_layer_param_t fw_broadcast_binary_layer_param = {0};

  dynamic_common_ir_layer_info(add_layer_info, getInputs()[0], getOutput());

  assign_fw_param((void *)&fw_broadcast_binary_layer_param);
  add_layer_info->fw_layer_param_u.fw_broadcast_binary_layer_param =
      fw_broadcast_binary_layer_param;
  fw_ir_length += sizeof(fw_broadcast_binary_layer_param_t);

  return fw_ir_length;
}

int64_t tpu::AddOp::get_fw_type_bm1684() { return FW_BMNET_BROADCAST_BINARY; }

// ======================================
// Dynamic LocalGenInterface
// ======================================

int32_t tpu::AddOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  int fw_ir_length = 0;
  ir_layer_info_t *add_layer_info = (ir_layer_info_t *)ir_layer_info;
  fw_broadcast_binary_layer_param_t fw_broadcast_binary_layer_param = {0};

  dynamic_common_ir_layer_info(add_layer_info, getInputs()[0], getOutput());

  assign_fw_param((void *)&fw_broadcast_binary_layer_param);
  add_layer_info->fw_layer_param_u.fw_broadcast_binary_layer_param =
      fw_broadcast_binary_layer_param;
  fw_ir_length += sizeof(fw_broadcast_binary_layer_param_t);

  // get layer input and output
  add_layer_info->ir_tensor_info_v.clear();
  // input tensor
  for (int i = 0; i < getInputs().size(); ++i) {
    dynamic_push_back_local_tensor(add_layer_info->ir_tensor_info_v,
                                   getInputs()[i]);
  }
  // output tensor
  dynamic_push_back_local_tensor(add_layer_info->ir_tensor_info_v, getOutput());

  if (add_layer_info->data_size != DSIZE_FP32) {
    dynamic_push_back_local_buffer(add_layer_info->ir_tensor_info_v,
                                   get_tensor_id(getInputs()[0]), getOutput());
  }

  // compute fw ir info length for input and output
  fw_ir_length += (uint32_t)(getInputs().size() + 1) * (2 * sizeof(uint32_t));

  if (add_layer_info->data_size != DSIZE_FP32)
    fw_ir_length += sizeof(uint32_t);

  // add fw ir length for output consumer number
  fw_ir_length += sizeof(uint32_t);

  return fw_ir_length;
}

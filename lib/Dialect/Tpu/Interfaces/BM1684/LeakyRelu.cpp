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

void tpu::LeakyReluOp::codegen_global_bm1684() {
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto bottom_global_addr = module::getAddress(getInput());
  auto top_global_addr = module::getAddress(getOutput());
  int channel_shared = 1;
  float slope_val = 0;
  if (module::isUniformQuantized(getInput())) {
    slope_val = static_cast<float>(getMultiplier().value());
    int rshift_bit = getRshift().value();
    int input_sign = module::isSign(getInput());
    int slope_sign = 1;
    int output_sign = module::isSign(getOutput());
    BM1684::instance().dl_nodechip_prelu_forward_fix8b(
        bottom_global_addr, 0, top_global_addr, slope_val, channel_shared, n, c,
        h, w, input_sign, slope_sign, output_sign, rshift_bit, 1, 1,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    slope_val = static_cast<float>(getAlpha().value().convertToDouble());
    BM1684::instance().dl_nodechip_prelu_forward(
        bottom_global_addr, 0, top_global_addr, slope_val, channel_shared, n, c,
        h, w, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

int64_t tpu::LeakyReluOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  n = in_nslice;
  h = in_hslice;
  auto NPU_NUM = BM1684::NPU_NUM;
  auto EU_NUM = BM1684::eu_num(sizeof(float));
  float slope_val = 0;
  int channel_shared = 1;
  int64_t buffer_size = 0;
  if (module::isUniformQuantized(getInput())) {
    slope_val = static_cast<float>(getMultiplier().value());
  } else {
    slope_val = static_cast<float>(getAlpha().value().convertToDouble());
  }
  bool need_buffer = !(channel_shared && slope_val == 0);
  if (need_buffer) {
    int64_t type_len = module::getDtypeSize(getInput());
    auto store_mode = type_len == 1 ? STORE_MODE_4N : STORE_MODE_1N;
    if (store_mode == STORE_MODE_4N && type_len == 1) {
      n = align_up(n, (int64_t)4);
    } else if (store_mode == STORE_MODE_2N && type_len == 2) {
      n = align_up(n, (int64_t)2);
    }
    int cnum = (c + NPU_NUM - 1) / NPU_NUM;
    int64_t cstride = h * w;
    cstride = align_up(cstride, (EU_NUM * 4 / type_len));
    buffer_size = n * cnum * cstride * type_len;
  }
  return buffer_size;
}

void tpu::LeakyReluOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                            local_sec_info_t &sec_info) {
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto out_gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  n = out_gi.n_slice;
  h = out_gi.h_slice;

  uint32_t la_input = in_gi.out_addr;
  uint32_t la_output = out_gi.out_addr;
  uint32_t la_buffer = out_gi.buffer_addr;
  int channel_shared = 1;
  float slope_val = 0;

  if (module::isUniformQuantized(getInput())) {
    slope_val = static_cast<float>(getMultiplier().value());
    int rshift_bit = getRshift().value();
    int upper_limit = -1;
    int input_sign = module::isSign(getInput());
    int slope_sign = 1;
    int output_sign = module::isSign(getOutput());

    uint32_t bottom_dim_fix8b[4] = {(uint32_t)n, (uint32_t)c, (uint32_t)h,
                                    (uint32_t)w};
    BM1684::instance().dl_nodechip_prelu_forward_local_fix8b_v3(
        la_input, la_output, 0, la_buffer, channel_shared, slope_val,
        bottom_dim_fix8b, 0, input_sign, slope_sign, output_sign, rshift_bit,
        upper_limit, (CMD_ID_NODE *)BM1684::instance()->bdc_node);
  } else {
    slope_val = static_cast<float>(getAlpha().value().convertToDouble());
    int bottom_dim[4] = {(int)n, (int)c, (int)h, (int)w};
    BM1684::instance().dl_nodechip_prelu_forward_local_v2(
        la_input, la_output, 0, la_buffer, channel_shared, slope_val,
        bottom_dim, 0, (CMD_ID_NODE *)BM1684::instance()->bdc_node);
  }
}

uint32_t tpu::LeakyReluOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  GLOBAL_IR_COMMON(prelu);
}

int64_t tpu::LeakyReluOp::get_fw_type_bm1684() { return FW_BMNET_PRELU; }

int32_t tpu::LeakyReluOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  int fw_ir_length = 0;
  IR_PARAM_COMMON(prelu);
  // input tensor
  dynamic_push_back_local_tensor(layer_info->ir_tensor_info_v, getInput());
  // weight no use
  // output
  dynamic_push_back_local_tensor(layer_info->ir_tensor_info_v, getOutput());
  // buffer
  bool need_buffer =
      !(layer_info->fw_layer_param_u.fw_prelu_layer_param.channel_shared &&
        layer_info->fw_layer_param_u.fw_prelu_layer_param.shared_slope == 0);
  int imm_length = 0;
  if (need_buffer) {
    dynamic_push_back_local_buffer(layer_info->ir_tensor_info_v, 0,
                                   getOutput());
    imm_length = 1;
  }
  // compute fw ir info length for lrn input and output
  if (layer_info->fw_layer_param_u.fw_prelu_layer_param.channel_shared) {
    fw_ir_length += (sizeof(uint32_t) + (2 + imm_length) * sizeof(uint32_t));
  } else {
    llvm_unreachable("not support");
  }
  fw_ir_length += sizeof(uint32_t);
  return fw_ir_length;
}

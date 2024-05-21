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

// =========================================
// GlobalGenInterface
// =========================================
void tpu::SwapChannelOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto channel_order = module::getI64Array(this->getChannelOrder());
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  swap_channel_param_t param = {0};
  param.shape_dim = 4;
  for (int i = 0; i < channel_order->size(); i++) {
    param.order[i] = channel_order->at(i);
  }
  BM168x::call_global_func("backend_api_swap_channel_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

int64_t tpu::SwapChannelOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(swap_channel_param_t);
  auto channel_order = module::getI64Array(this->getChannelOrder());
  swap_channel_param_t param = {0};
  param.shape_dim = 4;
  for (int i = 0; i < channel_order->size(); i++) {
    param.order[i] = channel_order->at(i);
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::SwapChannelOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return 0;
}

void tpu::SwapChannelOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                               int64_t h_step, int64_t d_step,
                                               int64_t w_step,
                                               group_type_t group_type,
                                               local_sec_info_t &sec_info) {
  UNREACHABLE_THIS("Not Implemented");
}

int64_t tpu::SwapChannelOp::get_fw_type_bm1684x() {
  return FW_BMNET_SWAP_CHANNEL;
}

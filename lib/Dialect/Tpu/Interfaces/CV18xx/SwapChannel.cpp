//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace tpu_mlir::backend;
// =========================================
// GlobalGenInterface
// =========================================

void tpu::SwapChannelOp::codegen_global_cv18xx(int64_t layer_id) {
  auto input_shape = module::getShape(this->getInput());
  std::vector<int> input_shape_fix(input_shape.begin(), input_shape.end());
  gaddr_t input_gaddr = module::getAddress(this->getInput());
  gaddr_t output_gaddr = module::getAddress(this->getOutput());
  auto channel_order = module::getI64Array(this->getChannelOrder());
  std::vector<int> order;
  for (int i = 0; i < channel_order->size(); i++) {
    order.push_back(channel_order->at(i));
  }
  if (module::isUniformQuantized(getOutput())) {
    cvi_backend_tg_swap_channel_kernel(
        layer_id, input_gaddr, output_gaddr, (int)input_shape_fix.size(),
        input_shape_fix.data(), order.data(), CVK_FMT_I8);
  } else {
    cvi_backend_tg_swap_channel_kernel(
        layer_id, input_gaddr, output_gaddr, (int)input_shape_fix.size(),
        input_shape_fix.data(), order.data(), CVK_FMT_BF16);
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::SwapChannelOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::SwapChannelOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                              int64_t d_step, int64_t w_step,
                                              group_type_t group_type,
                                              local_sec_info_t &sec_info,
                                              int64_t layer_id) {
  int64_t n, c, h, w;
  auto shape = module::getShape(getInput());
  module::getNCHW(shape, n, c, h, w);

  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto out_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);
  laddr_t la_input = in_gi.out_addr;
  laddr_t la_output = out_gi.out_addr;

  n = sec_info.n_slice;
  h = sec_info.h_slice;

  std::vector<int> order;
  auto channel_order = module::getI64Array(this->getChannelOrder());
  for (int i = 0; i < channel_order->size(); i++) {
    order.push_back(channel_order->at(i));
  }
  if (module::isUniformQuantized(getOutput())) {
    cvi_backend_tl_swap_channel(layer_id, la_input, la_output, n, c, h, w,
                                order.data(), CVK_FMT_I8);
  } else {
    cvi_backend_tl_swap_channel(layer_id, la_input, la_output, n, c, h, w,
                                order.data(), CVK_FMT_BF16);
  }
}

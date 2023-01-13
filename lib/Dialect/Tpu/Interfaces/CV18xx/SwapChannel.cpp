//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"




using namespace tpu_mlir::backend;
// =========================================
// GlobalGenInterface
// =========================================

void tpu::SwapChannelOp::codegen_global_cv18xx(int64_t layer_id) {

  //llvm_unreachable("Not supported now");
  std::vector<int64_t> input_shape;
  module::getShapeVec(this->getInput(), input_shape);
  std::vector<int> input_shape_fix;
  for (auto &dim : input_shape) {
    input_shape_fix.push_back((int)dim);
  }
  gaddr_t input_gaddr = module::getAddress(this->getInput());
  gaddr_t output_gaddr = module::getAddress(this->getOutput());
  auto channel_order = module::getI64Array(this->getChannelOrder());
  std::vector<int> order;
  for (int i = 0; i < channel_order->size(); i++) {
    order.push_back(channel_order->at(i));
  }
  if (module::isUniformQuantized(getOutput())) {
    cvi_backend_tg_swap_channel_kernel(layer_id,
                                      input_gaddr, output_gaddr,  (int)input_shape_fix.size(),
                                      input_shape_fix.data(), order.data(), CVK_FMT_I8);
  } else {
    cvi_backend_tg_swap_channel_kernel(layer_id,
                                      input_gaddr, output_gaddr,  (int)input_shape_fix.size(),
                                      input_shape_fix.data(), order.data(), CVK_FMT_BF16);
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::SwapChannelOp::getBufferSize_cv18xx(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes,
                                         int64_t in_nslice, int64_t in_hslice,
                                         int64_t out_nslice,
                                         int64_t out_hslice) {
  llvm_unreachable("Not supported now");
}

void tpu::SwapChannelOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step, int64_t layer_id) {
  llvm_unreachable("Not supported now");
}

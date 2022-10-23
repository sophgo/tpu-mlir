//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

void tpu::PReluOp::codegen_global_bm1684x() {
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  prelu_param_t p = {0};
  p.input_addr = Module::getAddress(input());
  p.output_addr = Module::getAddress(output());
  p.input_n = n;
  p.input_c = c;
  p.input_h = h;
  p.input_w = w;
  p.rshift_bit = rshift();
  p.relu_limit = -1;
  p.dtype = BM168x::getDataType(input());
  p.channel_shared = 0;
  p.slope_addr = Module::getAddress(slope());
  BM1684x::instance().call_global_func("backend_api_prelu_global", &p,
                                       sizeof(prelu_param_t));
}

int64_t tpu::PReluOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::PReluOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  prelu_param_t p = {0};
  p.input_addr = in_gi.out_addr;
  p.output_addr = gi.out_addr;
  p.input_n = static_cast<int32_t>(gi.n_slice);
  p.input_c = static_cast<int32_t>(c);
  p.input_h = static_cast<int32_t>(gi.h_slice);
  p.input_w = static_cast<int32_t>(w);
  p.rshift_bit = rshift();
  p.relu_limit = 0;
  p.dtype = BM168x::getDataType(input());
  auto slope_gi = LocalGenInterface::getGroupInfo(slope(), n_step, h_step);
  p.channel_shared = 0;
  p.slope_addr = slope_gi.out_addr;

  BM1684x::instance().call_local_func("backend_api_prelu_local", &p,
                                      sizeof(prelu_param_t));
}

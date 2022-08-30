//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::RequantIntAxisOp::codegen_global_bm1684x() {
  requant_int_param_t param = {0};
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  param.input_addr = Module::getAddress(input());
  param.requant_addr = Module::getAddress(quant());
  param.output_addr = Module::getAddress(output());
  param.n = (int)n;
  param.c = (int)c;
  param.h = (int)h;
  param.w = (int)w;

  param.is_perchannel = true;
  param.reshaped_coeff = false;
  param.mode = quant_mode();
  param.input_dtype = BM168x::getDataType(input());
  param.output_dtype = BM168x::getDataType(output());
  BM1684x::instance().call_global_func("backend_api_requant_int_global", &param,
                                       sizeof(param));
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::RequantIntAxisOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t buffer_size = 0;
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  if (quant_mode() == 0) {
    buffer_size = in_lmem_bytes;
    buffer_size += ceiling_func(c, BM1684x::NPU_NUM) * BM1684x::EU_BYTES;
  } else if (quant_mode() == 1) {
    buffer_size = in_lmem_bytes;
  }
  return buffer_size;
}

void tpu::RequantIntAxisOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  requant_int_param_t param = {0};
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto quant_gi = LocalGenInterface::getGroupInfo(quant(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  param.input_addr = (uint32_t)in_gi.out_addr;
  param.requant_addr = (uint32_t)quant_gi.out_addr;
  param.output_addr = (uint32_t)gi.out_addr;
  param.buffer_local_addr = (uint32_t)gi.buffer_addr;
  param.n = gi.n_slice;
  param.c = c;
  param.h = gi.h_slice;
  param.w = w;

  auto requant_gi = LocalGenInterface::getGroupInfo(quant(), n_step, h_step);
  param.requant_addr = (uint32_t)requant_gi.out_addr;
  param.is_perchannel = true;
  param.reshaped_coeff = false;

  if (Quant::isUniformQuantized(input())) {
    auto iqtype = Quant::getUniformQuantizedType(input());
    param.zx_value = iqtype.getZeroPoint();
  }
  param.input_dtype = BM168x::getDataType(input());
  param.output_dtype = BM168x::getDataType(output());
  param.mode = quant_mode();
  BM1684x::instance().call_local_func("backend_api_requant_int_local", &param,
                                      sizeof(param));
}

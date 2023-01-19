//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"



using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::RequantIntOp::codegen_global_bm1684x() {
  requant_int_param_t param = {0};
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  param.input_addr = module::getAddress(getInput());
  param.output_addr = module::getAddress(getOutput());
  param.n = (int)n;
  param.c = (int)c;
  param.h = (int)h;
  param.w = (int)w;

  auto oqtype = module::getUniformQuantizedType(getOutput());
  param.mul_value = getMultiplier();
  param.shift_value = -getRshift();
  param.offset_value = oqtype.getZeroPoint();
  if (module::isUniformQuantized(getInput())) {
    auto iqtype = module::getUniformQuantizedType(getInput());
    param.zx_value = iqtype.getZeroPoint();
  }
  param.mode = static_cast<int>(getQuantMode());
  param.input_dtype = BM168x::getDataType(getInput());
  param.output_dtype = BM168x::getDataType(getOutput());
  param.round_mode = getQuantMode() == tpu::RequantMode::MultiplierShift
                         ? ROUNDING_HALF_UP
                         : ROUNDING_HALF_AWAY_FROM_ZERO;
  BM168x::call_global_func("backend_api_requant_int_global", &param,
                           sizeof(param));
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::RequantIntOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t buffer_size = 0;
  auto input_dtype = BM168x::getDataType(getInput());
  if (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8) {
    // store INT16:(X - Zx)
    buffer_size = in_lmem_bytes * 2;
  } else if (getQuantMode() == tpu::RequantMode::TFLite_LShift ||
             getQuantMode() == tpu::RequantMode::TFLite) {
    buffer_size = in_lmem_bytes;
  }
  return buffer_size;
}

void tpu::RequantIntOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                              local_sec_info_t &sec_info) {
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto oqtype = module::getUniformQuantizedType(getOutput());

  requant_int_param_t param = {0};
  param.input_addr = (uint32_t)in_gi.out_addr;
  param.output_addr = (uint32_t)gi.out_addr;
  param.buffer_local_addr = (uint32_t)gi.buffer_addr;
  param.n = sec_info.n_slice;
  param.c = c;
  param.h = sec_info.h_slice;
  param.w = w;
  param.mul_value = getMultiplier();
  param.shift_value = -getRshift();
  param.offset_value = oqtype.getZeroPoint();

  if (module::isUniformQuantized(getInput())) {
    auto iqtype = module::getUniformQuantizedType(getInput());
    param.zx_value = iqtype.getZeroPoint();
  }
  param.input_dtype = BM168x::getDataType(getInput());
  param.output_dtype = BM168x::getDataType(getOutput());
  param.mode = static_cast<int>(getQuantMode());
  param.round_mode = getQuantMode() == tpu::RequantMode::MultiplierShift
                         ? ROUNDING_HALF_UP
                         : ROUNDING_HALF_AWAY_FROM_ZERO;
  BM168x::call_local_func("backend_api_requant_int_local", &param,
                          sizeof(param));
}

//dynamic codegen
int64_t tpu::RequantIntOp::dyn_codegen_local_bm1684x(void *buffer) {
return 0;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::RequantIntOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}
